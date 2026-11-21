"""Read-only instrumentation of complete AIS decisions and their search branches.

This module does not run a simulator or alter control selection. Lazy search
imports are installed from inside the measured decision branch. Nested branch
wall times overlap and must not be added as independent costs. Profiling adds
small Python/reporting overhead; independent unprofiled runs certify deadlines.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict
from functools import wraps
import importlib
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch


class ChooseBranchProfiler:
    def __init__(self):
        self.records = []
        self.constructors = []
        self.snapshots = []
        self._active = None
        self._stack = ExitStack()
        self._patched_modules = set()
        self._planner_constructors = {}

    @staticmethod
    def _cache(planner):
        module = importlib.import_module(type(planner).__module__)
        value = module._local_transition.cache_info()
        return dict(zip(('hits', 'misses', 'maxsize', 'currsize'), value))

    def _stat(self, name, seconds):
        if self._active is None:
            return
        item = self._active['branches'].setdefault(name, {'calls': 0, 'total_seconds': 0., 'max_seconds': 0.})
        item['calls'] += 1
        item['total_seconds'] += seconds
        item['max_seconds'] = max(item['max_seconds'], seconds)

    def _install_search_functions(self, branch):
        names = ['world.ais_clearance_refinement']
        if branch == '_recover_policy':
            names.append('world.ais_fast_recovery')
        for name in names:
            if name in self._patched_modules:
                continue
            # Import work remains inside the measured AIS choose/branch call.
            module = importlib.import_module(name)
            function = 'refine_clearance' if name.endswith('refinement') else 'search_feedback'
            original = getattr(module, function)

            @wraps(original)
            def measured(*args, _original=original, _function=function, **kwargs):
                start = perf_counter()
                result = None
                error = None
                try:
                    result = _original(*args, **kwargs)
                    return result
                except BaseException as caught:
                    error = repr(caught)
                    raise
                finally:
                    seconds = perf_counter() - start
                    self._stat(_function, seconds)
                    if self._active is not None and _function == 'refine_clearance':
                        self._active['nonlinear_refinements'].append({
                            'wall_seconds': seconds,
                            'max_iterations': kwargs.get('max_iterations', 180),
                            'cpu_seconds': kwargs.get('cpu_seconds', 8.),
                            'keep_best_iterate': kwargs.get('keep_best_iterate', False),
                            'returned_commands': result is not None,
                            'exception': error,
                        })
            self._stack.enter_context(patch.object(module, function, measured))
            self._patched_modules.add(name)

    def _branch(self, name, original):
        @wraps(original)
        def measured(planner, *args, **kwargs):
            start = perf_counter()
            try:
                if name in ('_recover_policy', '_refine_policy'):
                    self._install_search_functions(name)
                return original(planner, *args, **kwargs)
            finally:
                self._stat(name, perf_counter() - start)
        return measured

    def install(self, planner_class):
        original_constructor = planner_class.__init__

        @wraps(original_constructor)
        def constructor(planner, *args, **kwargs):
            start = perf_counter()
            try:
                return original_constructor(planner, *args, **kwargs)
            finally:
                event = {'planner_id': id(planner), 'constructor_seconds': perf_counter() - start}
                self.constructors.append(event)
                self._planner_constructors[id(planner)] = event

        self._stack.enter_context(patch.object(planner_class, '__init__', constructor))
        for name in ('_recover_policy', '_refine_policy', '_coarse_rollouts', '_rollout', '_screen', '_policies', '_refine_downwind_turns'):
            if not hasattr(planner_class, name):
                continue
            original = getattr(planner_class, name)
            self._stack.enter_context(patch.object(planner_class, name, self._branch(name, original)))
        original_choose = planner_class.choose

        @wraps(original_choose)
        def choose(planner, observation, grid, blocked, traffic, nominal_course, prefer_starboard):
            previous = self._active
            profile_before = getattr(planner, 'recovery_search_profile', None)
            record = {
                'attempt': len(self.records) + 1,
                'observation_utc': str(observation.time),
                'planner_id': id(planner),
                'nominal_course_deg': nominal_course,
                'prefer_starboard': bool(prefer_starboard),
                'target_count': len(traffic.targets),
                'cache_before': self._cache(planner),
                'retained_policy_before': planner.policy is not None,
                'retained_elapsed_seconds_before': planner.elapsed,
                'branches': {}, 'nonlinear_refinements': [],
            }
            self._active = record
            result = None
            error = None
            started = perf_counter()
            try:
                result = original_choose(planner, observation, grid, blocked, traffic, nominal_course, prefer_starboard)
                return result
            except BaseException as caught:
                error = repr(caught)
                raise
            finally:
                # Stop the complete choose timer before encoding diagnostic data.
                record['whole_choose_seconds'] = perf_counter() - started
                record['exception'] = error
                record['status'] = 'exception' if error else ('no_maneuver' if result is None else 'maneuver')
                record['cache_after'] = self._cache(planner)
                record['result'] = None if result is None else asdict(result)
                current_profile = getattr(planner, 'recovery_search_profile', None)
                if current_profile is not None and current_profile is not profile_before:
                    record['fresh_recovery_search_profile'] = dict(current_profile)
                record['fresh_recovery_attempted'] = '_recover_policy' in record['branches']
                record['ordinary_refinement_attempted'] = '_refine_policy' in record['branches']
                record['whole_choose_exceeds_5s'] = record['whole_choose_seconds'] >= 5.
                capture = (not self.records or error is not None or record['whole_choose_exceeds_5s']
                           or record['fresh_recovery_attempted'] and result is None
                           or result is not None and result.predicted_margin_metres <= 0.)
                if capture:
                    # Retain exact public observation/traffic and chart-mask input.
                    # This encoding is outside our timer, but inside any caller's
                    # timer around this profiling wrapper; that overhead is disclosed.
                    record['input_snapshot_index'] = len(self.snapshots)
                    self.snapshots.append({
                        'attempt': record['attempt'],
                        'observation': asdict(observation),
                        'grid': {'cells': grid.cells, 'resolution': grid.resolution,
                                 'span': grid.span, 'origin_latitude': grid.origin_latitude,
                                 'origin_longitude': grid.origin_longitude},
                        'blocked': blocked.copy(), 'traffic': asdict(traffic),
                        'nominal_course_deg': nominal_course, 'prefer_starboard': bool(prefer_starboard),
                    })
                self.records.append(record)
                self._active = previous
        self._stack.enter_context(patch.object(planner_class, 'choose', choose))
        return self

    def close(self):
        self._stack.close()

    def report(self):
        return {
            'scope': 'Complete AISBufferPlanner.choose including lazy search imports, native construction, search and validation; nested branch times overlap.',
            'timing_qualification': 'Diagnostic profiling overhead is present. Exact snapshot encoding occurs after this profiler timer but inside caller timing; independent unprofiled runs establish deadline results.',
            'no_actor_logic_changes': True,
            'constructor_events': self.constructors, 'decisions': self.records,
            'input_snapshots': self.snapshots,
        }


def save_report(path, payload):
    from world.navigation_replay import _encode_json
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Preserved profile already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, default=_encode_json, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

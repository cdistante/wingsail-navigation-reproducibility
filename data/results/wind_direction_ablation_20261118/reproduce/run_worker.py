"""Use study-local acceleration for navigation without changing production defaults."""

from contextlib import ExitStack
import sys

import run_wind_scenario


def main():
    with ExitStack() as stack:
        if "--reference-kernels" in sys.argv:
            sys.argv.remove("--reference-kernels")
        else:
            from fast_fossen import install as faster_prediction
            from reuse_fields import install as reuse_fields

            stack.enter_context(faster_prediction())
            stack.enter_context(reuse_fields())
        run_wind_scenario.main()


if __name__ == "__main__":
    main()

"""Correct study-level playback prose without changing frozen renderer receipts."""
from __future__ import annotations
import json
from pathlib import Path
STUDY=Path(__file__).resolve().parents[1]
def main():
    receipts=list((STUDY/'runs').glob('*/video_verification.json'))
    assert len(receipts)==8 and all((r:=json.loads(p.read_text()))['mission_status']=='completed' and len(r['arrivals'])==8 for p in receipts)
    path=STUDY/'playback.md';original=path.read_text();_,marker,table=original.partition('| Encounter |')
    assert marker and len([line for line in table.splitlines() if line.startswith('| ')])>=8
    prose='''# Otranto underway AIS encounter playback

All 8 loops completed all 64 ordered waypoint captures within the 50 m gates. The common stack is Gamma + Direct acados NMPC with regatta time-to-mark guidance and refined forecast tacks on both outbound and return legs. The Transformer treatment adds the frozen weather hold/release model and physical confirmation. AIS intervention and nominal weather guidance are identified separately in every clip.

Requested departure is 2026-09-30 23:50 UTC. Four opening seconds summarize the recorded protected-staging release certificate over the actual launch fix; no prelaunch in-water trajectory is invented. Each clip labels actual launch and sheltered wait separately. Original absolute traffic and report-fault UTC anchors remain fixed.

Reports retain a 20 s delivery delay, two 40 s staged emission gaps and two 40 s underway emission gaps, with a 65 s planning age budget. Both actual 65 s held-age peaks in the later encounter were experienced while underway in every arm. Applied checks use uncapped actual report age. The nominal 25 m gap, full uncertainty allowance and additional 3 m reserve remain distinct checks.

All loops share a fixed north-up overview. A tactical inset shows the later encounter, retaining every recorded update from original traffic epoch 420–900 s. Red domains belong only to received reports at the displayed fix; used-report age and step margins refer to the control interval ending there. Gray diamonds before acquisition show physical truth for offline audit/display, never a controller input. Purple hollow circles show truth for reported identities. Exact camera bounds, enlarged boat scale and retained source indices are in each video receipt.

True blue 50 m disks and orange capture dots identify the waypoint gates and their actual entries. The yellow dot marks the bow, and the purple arrow shows measured COG; signed surge labels distinguish sternway. Wind barbs show FROM and the green current arrow shows TO. Orange nominal aims remain contained within the unchanged true gate on either leg.

The 5 s AIS budget is checked against complete choose-through-acceptance computation; the upstream last regatta-search cost has a separate scope and can repeat for cached forecasts. All clips contain 508 frames at 12 fps, including the four-second launch summary and half-second capture holds. Every capture, its neighboring fixes, all recorded later-encounter updates, COLREG/source changes and final state are retained. Full raw replays preserve every five-second update; only video playback is downsampled.

'''
    path.write_text(prose+marker+table)
    assert path.read_text().partition('| Encounter |')[2]==table
    print(path)
if __name__=='__main__':main()

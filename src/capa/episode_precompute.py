"""
CAPA Episode EIM Precompute

Efficiently precompute EIM state vectors (EIM_t) for unified CAPA episodes
by walking the transcript once per meeting and capturing state at cue points.
"""

from typing import List, Dict, Any, Optional

from schemas import CAPAEpisode, EIMMeetingStateVector
from environment_interpreter import EnvironmentInterpretationModule


def _build_utt_index(transcript: List[Dict[str, Any]]) -> Dict[str, int]:
    idx = {}
    for i, utt in enumerate(transcript):
        uid = utt.get("utterance_id")
        if uid:
            idx[uid] = i
    return idx


def precompute_eim_for_episodes(
    transcript: List[Dict[str, Any]],
    episodes: List[CAPAEpisode],
    eim: EnvironmentInterpretationModule,
) -> List[Dict[str, Any]]:
    """
    Precompute EIMMeetingStateVector (EIM_t) for each episode cue efficiently.

    Walk the transcript once and, in ascending cue order, process new utterances
    into EIM.master_state, then call analyze_snapshot(context_last_N) to capture
    the full EIMMeetingStateVector for that cue t.

    Returns a list of { "test_case_id": str, "eim_output": <EIMMeetingStateVector as dict> }.
    """
    if not transcript or not episodes:
        return []

    # Map utterance_id to index for ordering and validation
    utt_to_idx = _build_utt_index(transcript)

    # Sort episodes by cue index (prefer numeric cue_index, fallback to utt_id mapping)
    def _cue_index(ep: CAPAEpisode) -> int:
        if ep.cue_index is not None:
            return int(ep.cue_index)
        return utt_to_idx.get(ep.cue_utt_id, 10**9)

    episodes_sorted = sorted(episodes, key=_cue_index)

    outputs: List[Dict[str, Any]] = []
    last_processed = -1

    for ep in episodes_sorted:
        t = _cue_index(ep)
        if t >= len(transcript):
            # skip invalid cues
            continue
        # Process new history up to t-N (optional) or directly to t; EIM's history build is tolerant.
        # We process utterances from (last_processed+1) .. t-N to build context, then analyze the snapshot.
        hist_end = max(0, t - (ep.N - 1))
        if hist_end - 1 > last_processed:
            # Add history before the context window
            eim.process_history(transcript[last_processed + 1: hist_end])
            last_processed = hist_end - 1

        # Now, the context window itself should be analyzed as a snapshot to produce EIM_t with fresh snapshot_analysis.
        context = ep.context_last_N
        vec = eim.analyze_snapshot(context)
        if vec is None:
            continue
        # Capture the full EIMMeetingStateVector (historical WM + snapshot)
        outputs.append({
            "test_case_id": ep.test_case_id or f"{ep.meeting_id}_{ep.represented_user}_t{t}",
            "eim_output": vec.model_dump(),
        })
    return outputs


def write_eim_states_to_file(states: List[Dict[str, Any]], output_path: str) -> None:
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(states, f, indent=2, ensure_ascii=False)


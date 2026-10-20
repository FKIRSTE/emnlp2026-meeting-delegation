"""
CAPA Episode Generator (Unified): builds per-user episodes with context_last_N and
gt_next_window suitable for both environment and delegate judging.

This generator does not use the LLM; it creates dense episodes deterministically.
Might later add an LLM-based filter to select focused episodes, if needed.
"""

from typing import List, Dict, Any, Optional
from schemas import CAPAEpisode


def _derive_meeting_id(utt_id: str) -> str:
    try:
        return str(utt_id).split('_')[0]
    except Exception:
        return "Meeting"


def generate_dense_episodes(
    transcript: List[Dict[str, Any]],
    represented_user: str,
    N: int = 15,
    k: int = 5,
    step: int = 10,
) -> List[CAPAEpisode]:
    """
    Build dense, per-cue CAPA episodes for a given represented user. This creates
    an episode at every cue t where there is enough history and future window.

    Args:
        transcript: list of utterance dicts with keys including 'utterance_id', 'speaker_id', 'utterance_content'
        represented_user: participant id/role (for delegate judging)
        N: history window size 
        k: future window size
        step: stride for cues (1 means every utterance)

    Returns:
        List of CAPAEpisode objects
    """
    episodes: List[CAPAEpisode] = []
    if not transcript:
        return episodes

    total = len(transcript)
    # t is the index of the cue utterance
    start = max(0, N - 1)
    end = total - k - 1  # ensure t+k exists
    if end < start:
        return episodes

    for t in range(start, end + 1, step):
        context = transcript[t - (N - 1): t + 1]
        gt_window = transcript[t + 1: t + 1 + k]
        if len(context) < N or len(gt_window) < k:
            continue
        cue_utt_id = context[-1].get('utterance_id', f"t_{t}")
        meeting_id = _derive_meeting_id(cue_utt_id)
        episode = CAPAEpisode(
            meeting_id=meeting_id,
            represented_user=represented_user,
            cue_utt_id=cue_utt_id,
            cue_index=t,
            N=N,
            k=k,
            context_last_N=context,
            gt_next_window=gt_window,
            test_case_id=f"{meeting_id}_{represented_user}_t{t}"
        )
        episodes.append(episode)
    return episodes


def generate_dense_episodes_multi_participant(
    transcript: List[Dict[str, Any]],
    participants: List[str],
    N: int = 12,
    k: int = 5,
    step: int = 1,
) -> Dict[str, List[CAPAEpisode]]:
    """
    OPTIMIZED: Generate dense episodes once, then clone for multiple participants.
    
    This avoids redundant sliding window computation by generating the base episodes
    once and then creating participant-specific copies by only changing the
    represented_user and test_case_id fields.
    
    Args:
        transcript: list of utterance dicts
        participants: list of participant ids (e.g., ["PM", "Designer", "Dev"])
        N: history window size
        k: future window size
        step: stride for cues
        
    Returns:
        Dict mapping participant_id to their episode list
    """
    if not participants:
        return {}
    
    # Generate base episodes once (using first participant as template)
    base_episodes = generate_dense_episodes(
        transcript=transcript,
        represented_user=participants[0],
        N=N,
        k=k,
        step=step
    )
    
    result: Dict[str, List[CAPAEpisode]] = {}
    
    # First participant gets the original
    result[participants[0]] = base_episodes
    
    # For remaining participants, clone episodes with only represented_user changed
    for participant in participants[1:]:
        cloned_episodes = []
        for base_ep in base_episodes:
            # Create new episode with same data but different represented_user
            cloned_ep = CAPAEpisode(
                meeting_id=base_ep.meeting_id,
                represented_user=participant,  # ← Only this changes
                cue_utt_id=base_ep.cue_utt_id,
                cue_index=base_ep.cue_index,
                N=base_ep.N,
                k=base_ep.k,
                context_last_N=base_ep.context_last_N,  # Shared reference (immutable)
                gt_next_window=base_ep.gt_next_window,  # Shared reference (immutable)
                test_case_id=f"{base_ep.meeting_id}_{participant}_t{base_ep.cue_index}"
            )
            cloned_episodes.append(cloned_ep)
        result[participant] = cloned_episodes
    
    return result


def write_episodes_to_file(episodes: List[CAPAEpisode], output_path: str) -> None:
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([e.model_dump() for e in episodes], f, indent=2, ensure_ascii=False)


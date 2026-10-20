"""
CAPA Episode Runner

Runs a unified CAPA episode (per-user) over k steps (δ-loop) using the
Two-Tier agent and CAPA driver. At each step δ, it:
  - passes a shrinking gt_window = gt_next_window[δ:]
  - anchors patches to gt_window[0] (u(t+δ+1)) via the driver
  - advances the snapshot window by appending the observed gt turn

This module is leakage-safe: it never injects future turns into EIM except via
strict, current-turn-anchored patches.
"""

from typing import List, Dict, Optional, Any

from schemas import (
    CAPAEpisode,
    EIMMeetingStateVector,
    EIMMasterState,
)
from shared_memory import SharedMemoryModule
from environment_interpreter import EnvironmentInterpretationModule
from two_tier_agent import TwoTierUserModule


def _ensure_eim_for_cue(
    eim: EnvironmentInterpretationModule,
    smm: SharedMemoryModule,
    context_last_N: List[Dict[str, Any]],
) -> None:
    """
    Ensure SMM has a working EIM state for the cue built only from context_last_N.
    If SMM already has working memory, reuse it; otherwise, analyze the snapshot
    from an empty master_state and set it as working memory.
    """
    if smm.get_working_memory() is not None:
        return
    vec = eim.analyze_snapshot(context_last_N)
    if vec is not None:
        smm.update_working_memory(vec)


def run_capa_episode(
    agent: TwoTierUserModule,
    smm: SharedMemoryModule,
    episode: CAPAEpisode,
    participant_id: str,
    eim_t: Optional[EIMMeetingStateVector] = None,
    eim: Optional[EnvironmentInterpretationModule] = None,
) -> List[Dict[str, Any]]:
    """
    Run a CAPA episode over k steps (δ-loop).

    Args:
        agent: TwoTierUserModule instance (with CAPA enabled via config if desired)
        smm: SharedMemoryModule instance
        episode: CAPAEpisode containing context_last_N and gt_next_window
        participant_id: represented user id/role for the delegate
        eim_t: Optional EIMMeetingStateVector for cue t; if None, will be computed
               from context_last_N using a provided or transient EIM instance
        eim: Optional EnvironmentInterpretationModule to compute EIM when eim_t is None

    Returns:
        List of per-step records (dict) with delegate_response and anchors used.
    """
    records: List[Dict[str, Any]] = []

    # Seed EIM working memory at cue t
    if eim_t is not None:
        smm.update_working_memory(eim_t)
        try:
            smm.mark_wm_seeded_at_cue()
        except Exception:
            pass
    else:
        _eim = eim or EnvironmentInterpretationModule(agent.client)
        _ensure_eim_for_cue(_eim, smm, episode.context_last_N)
        try:
            smm.mark_wm_seeded_at_cue()
        except Exception:
            pass

    # Maintain the snapshot window locally, start with context_last_N
    snapshot_window: List[Dict[str, Any]] = list(episode.context_last_N)

    # δ-loop over k steps
    for delta in range(episode.k):
        gt_window = episode.gt_next_window[delta:]
        if not gt_window:
            break
        # Run a single turn; CAPA will patch EIM using gt_window[0] as anchor
        delegate_response = agent.run_turn(
            smm=smm,
            snapshot_utterances=snapshot_window,
            participant_id=participant_id,
            gt_next_window=gt_window,
        )

        # Advance snapshot window by appending the observed turn and trimming to N
        observed_turn = gt_window[0]
        snapshot_window = list(snapshot_window) + [observed_turn]
        # Trim to length N (episode.N)
        if len(snapshot_window) > episode.N:
            snapshot_window = snapshot_window[-episode.N:]

        records.append({
            "delta": delta,
            "anchor_utt_id": observed_turn.get("utterance_id"),
            "delegate_response": delegate_response.model_dump() if delegate_response else None,
        })

    return records

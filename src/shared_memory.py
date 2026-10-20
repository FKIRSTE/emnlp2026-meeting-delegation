"""
Shared Memory Module (SMM)

This module implements the central "mind" of the delegate agent, providing
a robust container and state manager for all memory components. It maintains
architectural decoupling and serves as the single source of truth for agent state.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

# Add current directory to path for robust importing
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from schemas import UserProfile, EIMMeetingStateVector, DelegateActionLogEntry
from capa.types import EpisodicEvent, PendingItem
from capa.episodic import PendingQueue


class SharedMemoryModule:
    """
    The Shared Memory Module (SMM) acts as the central, structured "mind" of the delegate agent.
    
    This module is not intelligent; it is a robust container and state manager that provides
    a clean API for other modules (EIM, UM) to read from and write to, ensuring architectural
    decoupling and a single source of truth for the agent's state.
    
    Memory Components:
    1. Long-Term Memory: User's stable identity and profile
    2. JIT Briefing Memory: Session-specific instructions and knowledge  
    3. Working Memory: Live perception of the meeting environment (from EIM)
    4. Episodic Memory: Agent's memory of its own actions in this meeting
    """
    
    def __init__(self):
        """
        Initialize the SMM with empty memory components.
        """
        # 1. The user's stable, long-term identity
        self.long_term_memory: Optional[UserProfile] = None

        # 2. The specific instructions/knowledge for THIS meeting session
        self.jit_briefing_memory: List[str] = []

        # 3. The agent's perception of the live, evolving environment from the EIM
        self.working_memory: Optional[EIMMeetingStateVector] = None

        # 4. The agent's memory of its own actions in this meeting
        self.episodic_memory: List[DelegateActionLogEntry] = []

        # CAPA episodic audit trail (predictions, error reports, patches)
        self.capa_events: List[EpisodicEvent] = []

        # CAPA pending queue for aligned-late recognition (optional use)
        self.pending_queue: PendingQueue = PendingQueue(ttl_k=5)

        # Optional recalibration guidance (scaffold)
        self.recalibration_guidance: Optional[Dict[str, Any]] = None

        # Internal flag: indicates WM snapshot at cue was pre-seeded (avoid redundant refresh once)
        self._wm_seeded_at_cue: bool = False

        # Episode-level tracking for recalibration guidance
        self.matched_idea_units: Set[str] = set()  # Accumulated matched unit IDs
        self.episode_contribution_count: int = 0   # How many times delegate spoke this episode
        self.last_contribution_step: Optional[int] = None  # For cooldown detection
        
        print("SMM Initialized: All memory components are empty.")

    # --- METHODS FOR WRITING TO MEMORY ---

    def load_user_profile(self, profile: UserProfile):
        """
        Load the static, long-term user profile.
        
        Args:
            profile: UserProfile instance containing user's stable identity
        """
        self.long_term_memory = profile
        print(f"  [SMM] Long-Term Memory loaded for user.")

    def load_jit_briefing(self, briefing_points: List[str]):
        """
        Load the specific talking points for the current session.
        
        Args:
            briefing_points: List of session-specific briefing points
        """
        self.jit_briefing_memory = briefing_points
        print(f"  [SMM] JIT Briefing Memory loaded with {len(briefing_points)} points.")

    def update_working_memory(self, meeting_state: EIMMeetingStateVector):
        """
        Update the agent's perception of the world with the latest from the EIM.
        
        Args:
            meeting_state: Latest meeting state vector from Environment Interpretation Module
        """
        self.working_memory = meeting_state
        print(f"  [SMM] Working Memory updated to state at utterance '{meeting_state.last_processed_utterance_id}'.")

    # --- CAPA helper flags ---
    def mark_wm_seeded_at_cue(self) -> None:
        self._wm_seeded_at_cue = True

    def consume_wm_seeded_flag(self) -> bool:
        """Return and clear the 'seeded at cue' flag (one-shot)."""
        val = self._wm_seeded_at_cue
        self._wm_seeded_at_cue = False
        return val

    def log_delegate_action(self, action_entry: DelegateActionLogEntry):
        """
        Log an action taken by the User Module to the agent's own memory.
        
        Args:
            action_entry: DelegateActionLogEntry containing details of the action taken
        """
        self.episodic_memory.append(action_entry)
        print(f"  [SMM] Logged new action to Episodic Memory: '{action_entry.action_taken}'.")

    # --- METHODS FOR READING FROM MEMORY (The API for the User Module) ---

    def get_user_profile(self) -> Optional[UserProfile]:
        """
        Retrieve the user's long-term profile.
        
        Returns:
            UserProfile instance or None if not loaded
        """
        return self.long_term_memory

    def get_jit_briefing(self) -> List[str]:
        """
        Retrieve the session-specific briefing points.
        
        Returns:
            List of briefing points
        """
        return self.jit_briefing_memory

    def get_working_memory(self) -> Optional[EIMMeetingStateVector]:
        """
        Retrieve the current meeting state from working memory.
        
        Returns:
            Current EIMMeetingStateVector or None if not available
        """
        return self.working_memory

    def get_episodic_memory(self) -> List[DelegateActionLogEntry]:
        """
        Retrieve the agent's memory of its own actions.
        
        Returns:
            List of DelegateActionLogEntry instances
        """
        return self.episodic_memory

    # --- CAPA episodic & pending APIs ---

    def log_capa_event(self, event: EpisodicEvent) -> None:
        """Record a CAPA episodic event (prediction, error, patch summary)."""
        self.capa_events.append(event)

    def get_capa_events(self) -> List[EpisodicEvent]:
        """Retrieve CAPA episodic events."""
        return self.capa_events

    def clear_capa_events(self) -> None:
        """Clear CAPA episodic events."""
        self.capa_events = []

    def pending_insert(self, item: PendingItem) -> None:
        """Insert a pending prediction item (optional signature)."""
        self.pending_queue.insert(item)

    def pending_fulfill(self, signature: Optional[str], matched_utt_id: Optional[str] = None) -> bool:
        """Mark a pending item as fulfilled by signature, if available."""
        return self.pending_queue.fulfill(signature, matched_utt_id)

    def pending_expire(self, signature: Optional[str]) -> bool:
        """Expire a pending item by signature, if available."""
        return self.pending_queue.expire(signature)

    def pending_decay(self) -> None:
        """Decay pending items TTL and remove expired ones."""
        self.pending_queue.decay()

    # --- Episode-level tracking for recalibration guidance ---

    def record_idea_unit_match(self, unit_id: str) -> None:
        """Record that an idea unit was matched by delegate contribution."""
        if unit_id:
            self.matched_idea_units.add(unit_id)

    def record_delegate_contribution(self, step: int, delta_index: int = 0) -> None:
        """Record that delegate contributed (matched) at this step and delta."""
        self.episode_contribution_count += 1
        self.last_contribution_step = step
        self.last_contribution_delta = delta_index  # Track delta for within-episode cooldown

    def is_in_cooldown(self, current_step: int, cooldown_steps: int = 2) -> bool:
        """Check if delegate recently contributed (within cooldown window by step count)."""
        if self.last_contribution_step is None:
            return False
        return (current_step - self.last_contribution_step) <= cooldown_steps

    def is_in_cooldown_delta(self, current_delta: int, cooldown_deltas: int = 2) -> bool:
        """Check if delegate recently contributed (within cooldown window by delta index).
        
        This is the preferred method for CAPA simulations where delta represents
        sequential turns within an episode. Cooldown expires after cooldown_deltas steps.
        """
        if not hasattr(self, 'last_contribution_delta') or self.last_contribution_delta is None:
            return False
        return (current_delta - self.last_contribution_delta) <= cooldown_deltas

    def get_idea_unit_coverage(self, total_units: int) -> Tuple[int, int]:
        """Get (matched_count, total_count) for idea unit coverage."""
        return (len(self.matched_idea_units), total_units)

    def reset_episode_tracking(self) -> None:
        """Reset episode-level tracking (call at episode start)."""
        self.matched_idea_units = set()
        self.episode_contribution_count = 0
        self.last_contribution_step = None
        self.last_contribution_delta = None  # Reset delta tracking

    def get_full_state_as_dict(self) -> Dict[str, Any]:
        """
        Create a complete, serializable snapshot of the SMM's current state.
        This is essential for logging and debugging.
        
        Returns:
            Dictionary containing complete SMM state
        """
        state = {
            "long_term_memory": self.long_term_memory.model_dump() if self.long_term_memory else None,
            "jit_briefing_memory": self.jit_briefing_memory,
            "working_memory": self.working_memory.model_dump() if self.working_memory else None,
            "episodic_memory": [entry.model_dump() for entry in self.episodic_memory],
        }
        # Include CAPA episodic and pending snapshot if present
        try:
            state["capa_events"] = [e.model_dump() for e in self.capa_events]
        except Exception:
            state["capa_events"] = []
        try:
            state["pending_queue"] = [it.model_dump() for it in self.pending_queue.items]
        except Exception:
            state["pending_queue"] = []
        try:
            state["recalibration_guidance"] = self.recalibration_guidance
        except Exception:
            state["recalibration_guidance"] = None
        return state

    def clear_episodic_memory(self):
        """
        Clear the episodic memory (useful for new meetings/sessions).
        """
        self.episodic_memory = []
        print("  [SMM] Episodic Memory cleared.")

    def clear_working_memory(self):
        """
        Clear the working memory (useful for meeting transitions).
        """
        self.working_memory = None
        print("  [SMM] Working Memory cleared.")

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current memory state for debugging.
        
        Returns:
            Dictionary with memory component summaries
        """
        return {
            "has_long_term_memory": self.long_term_memory is not None,
            "jit_briefing_points_count": len(self.jit_briefing_memory),
            "has_working_memory": self.working_memory is not None,
            "episodic_memory_entries": len(self.episodic_memory),
            "capa_events_entries": len(getattr(self, 'capa_events', []) or []),
            "pending_queue_items": len(getattr(self, 'pending_queue', PendingQueue()).items or []),
            "last_processed_utterance": (
                self.working_memory.last_processed_utterance_id 
                if self.working_memory else None
            )
        }

    def get_memory_state_summary(self) -> str:
        """
        Get a formatted string summary of the current memory state.
        
        Returns:
            String with memory component status
        """
        ltm_status = "LOADED" if self.long_term_memory else "EMPTY"
        wm_status = "LOADED" if self.working_memory else "EMPTY"
        jit_count = len(self.jit_briefing_memory)
        episodic_count = len(self.episodic_memory)
        capa_count = len(getattr(self, 'capa_events', []) or [])
        pending_count = len(getattr(self, 'pending_queue', PendingQueue()).items or [])

        return (
            f"Long-Term Memory: {ltm_status}, Working Memory: {wm_status}, "
            f"JIT Briefing: {jit_count} points, Episodic Memory: {episodic_count} entries, "
            f"CAPA Events: {capa_count}, Pending: {pending_count}"
        )

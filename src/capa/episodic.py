"""
CAPA Pending Queue scaffolding.
"""

from typing import List, Optional
from .types import PendingItem


class PendingQueue:
    def __init__(self, ttl_k: int = 5):
        self.ttl_k = ttl_k
        self.items: List[PendingItem] = []

    def insert(self, item: PendingItem) -> None:
        self.items.append(item)

    def fulfill(self, signature: Optional[str], matched_utt_id: Optional[str] = None) -> bool:
        for i, it in enumerate(self.items):
            if signature is not None and it.intent_signature == signature:
                it.matched_utt_id = matched_utt_id
                del self.items[i]
                return True
        return False

    def expire(self, signature: Optional[str]) -> bool:
        for i, it in enumerate(self.items):
            if signature is not None and it.intent_signature == signature:
                del self.items[i]
                return True
        return False

    def decay(self) -> None:
        # Simple linear decay toward expiry; remove when ttl_left hits zero.
        survivors: List[PendingItem] = []
        for it in self.items:
            if it.ttl_left is not None:
                it.ttl_left -= 1
                if it.ttl_left <= 0:
                    continue
            survivors.append(it)
        self.items = survivors


__all__ = ["PendingQueue"]

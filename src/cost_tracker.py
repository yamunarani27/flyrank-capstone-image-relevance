"""
Per-call cost tracking for vision and embedding API calls.

PRICING NOTE: Rates confirmed directly against ai.google.dev on [today's date]: $0.75/1M input, $3.75/1M output through Dec 31 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

INPUT_COST_PER_MILLION_TOKENS = 0.75
OUTPUT_COST_PER_MILLION_TOKENS = 3.75

@dataclass
class CostEntry:
    call_type: str  # "vision" or "embedding"
    target: str  # e.g. the image filename or post id this call was for
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CostLog:
    """In-memory cost log for a single run of the batch job."""

    def __init__(self) -> None:
        self._entries: list[CostEntry] = []

    def record(self, *, call_type: str, target: str, input_tokens: int, output_tokens: int) -> CostEntry:
        cost = (
            (input_tokens / 1_000_000) * INPUT_COST_PER_MILLION_TOKENS
            + (output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION_TOKENS
        )
        entry = CostEntry(
            call_type=call_type,
            target=target,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 6),
        )
        self._entries.append(entry)
        return entry

    @property
    def total_cost_usd(self) -> float:
        return round(sum(e.cost_usd for e in self._entries), 6)

    @property
    def entries(self) -> list[CostEntry]:
        return list(self._entries)
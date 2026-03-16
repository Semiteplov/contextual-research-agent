from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations
from typing import Any

from contextual_research_agent.retrieval.config import FusionConfig
from contextual_research_agent.retrieval.types import (
    ChannelResult,
    FusionResult,
    ScoredCandidate,
)


class FusionStrategy(ABC):
    @abstractmethod
    def fuse(
        self,
        channel_results: list[ChannelResult],
        top_n: int = 50,
        **kwargs: Any,
    ) -> FusionResult:
        """
        Merge candidates from multiple channels into a single ranked list.

        Args:
            channel_results: Per-channel retrieval results.
            top_n: Maximum candidates to retain after fusion.

        Returns:
            FusionResult with merged candidates and contribution stats.
        """
        ...


class ReciprocalRankFusion(FusionStrategy):
    """
    Reciprocal Rank Fusion.
    RRF_score(d) = Σ_c  w_c / (k + rank_c(d))
    """

    def __init__(self, config: FusionConfig | None = None):
        self._config = config or FusionConfig()

    def fuse(
        self,
        channel_results: list[ChannelResult],
        top_n: int | None = None,
        **kwargs: Any,
    ) -> FusionResult:
        top_n = top_n or self._config.top_n
        k = self._config.rrf_k
        weights = self._config.channel_weights

        t0 = time.perf_counter()

        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, ScoredCandidate] = {}
        chunk_channels: dict[str, set[str]] = defaultdict(set)

        for cr in channel_results:
            channel_weight = weights.get(cr.channel.value, 1.0)

            for candidate in cr.candidates:
                cid = candidate.chunk_id

                rrf_scores[cid] += channel_weight / (k + candidate.rank)

                if cid not in chunk_map or candidate.score > chunk_map[cid].score:
                    chunk_map[cid] = candidate

                chunk_channels[cid].add(cr.channel.value)

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_ids = sorted_ids[:top_n]

        fused: list[ScoredCandidate] = []
        for rank, cid in enumerate(top_ids):
            original = chunk_map[cid]
            fused.append(
                ScoredCandidate(
                    chunk=original.chunk,
                    score=rrf_scores[cid],
                    rank=rank,
                    channel=original.channel,
                    metadata={
                        **original.metadata,
                        "rrf_score": rrf_scores[cid],
                        "original_score": original.score,
                        "source_channels": list(chunk_channels[cid]),
                    },
                )
            )

        contributions: dict[str, int] = defaultdict(int)
        for cid in top_ids:
            for ch in chunk_channels[cid]:
                contributions[ch] += 1

        overlaps: dict[str, int] = {}
        active_channels = [cr.channel.value for cr in channel_results]
        for ch_a, ch_b in combinations(active_channels, 2):
            pair_key = f"{ch_a}∩{ch_b}"
            count = sum(
                1 for cid in top_ids if ch_a in chunk_channels[cid] and ch_b in chunk_channels[cid]
            )
            if count > 0:
                overlaps[pair_key] = count

        latency_ms = (time.perf_counter() - t0) * 1000

        return FusionResult(
            candidates=fused,
            channel_contributions=dict(contributions),
            channel_overlaps=overlaps,
            latency_ms=latency_ms,
        )


class WeightedScoreFusion(FusionStrategy):
    """
    Weighted linear combination of normalized scores.
    Score_fused(d) = Σ_c  w_c × norm_score_c(d)
    """

    def __init__(self, config: FusionConfig | None = None):
        self._config = config or FusionConfig()

    def fuse(  # noqa: C901
        self,
        channel_results: list[ChannelResult],
        top_n: int | None = None,
        **kwargs: Any,
    ) -> FusionResult:
        top_n = top_n or self._config.top_n
        weights = self._config.channel_weights
        t0 = time.perf_counter()

        normalized: dict[str, dict[str, float]] = {}
        chunk_map: dict[str, ScoredCandidate] = {}
        chunk_channels: dict[str, set[str]] = defaultdict(set)

        for cr in channel_results:
            scores = [c.score for c in cr.candidates]
            if not scores:
                continue
            min_s, max_s = min(scores), max(scores)
            span = max_s - min_s if max_s > min_s else 1.0

            channel_norm: dict[str, float] = {}
            for c in cr.candidates:
                norm = (c.score - min_s) / span
                channel_norm[c.chunk_id] = norm

                if c.chunk_id not in chunk_map or c.score > chunk_map[c.chunk_id].score:
                    chunk_map[c.chunk_id] = c

                chunk_channels[c.chunk_id].add(cr.channel.value)

            normalized[cr.channel.value] = channel_norm

        fused_scores: dict[str, float] = defaultdict(float)
        for channel_name, channel_scores in normalized.items():
            w = weights.get(channel_name, 1.0)
            for cid, norm_score in channel_scores.items():
                fused_scores[cid] += w * norm_score

        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        top_ids = sorted_ids[:top_n]

        fused: list[ScoredCandidate] = []
        for rank, cid in enumerate(top_ids):
            original = chunk_map[cid]
            fused.append(
                ScoredCandidate(
                    chunk=original.chunk,
                    score=fused_scores[cid],
                    rank=rank,
                    channel=original.channel,
                    metadata={
                        **original.metadata,
                        "fused_score": fused_scores[cid],
                        "original_score": original.score,
                        "source_channels": list(chunk_channels[cid]),
                    },
                )
            )

        contributions: dict[str, int] = defaultdict(int)
        for cid in top_ids:
            for ch in chunk_channels[cid]:
                contributions[ch] += 1

        overlaps: dict[str, int] = {}
        active = [cr.channel.value for cr in channel_results]
        for ch_a, ch_b in combinations(active, 2):
            pair_key = f"{ch_a}∩{ch_b}"
            count = sum(
                1 for cid in top_ids if ch_a in chunk_channels[cid] and ch_b in chunk_channels[cid]
            )
            if count > 0:
                overlaps[pair_key] = count

        return FusionResult(
            candidates=fused,
            channel_contributions=dict(contributions),
            channel_overlaps=overlaps,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


def create_fusion_strategy(config: FusionConfig | None = None) -> FusionStrategy:
    config = config or FusionConfig()
    if config.strategy == "rrf":
        return ReciprocalRankFusion(config)
    if config.strategy == "weighted":
        return WeightedScoreFusion(config)
    raise ValueError(f"Unknown fusion strategy: {config.strategy}")

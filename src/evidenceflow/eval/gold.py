"""Chunk-ID Gold coverage and rank computation shared across experiments."""

from collections.abc import Sequence


def _validated_depths(depths: Sequence[int]) -> tuple[int, ...]:
    """Normalize positive-integer depths and reject duplicate keys that would let metrics overwrite each other."""

    normalized = tuple(depths)
    if not normalized or any(type(depth) is not int or depth <= 0 for depth in normalized):
        raise ValueError("depths must contain positive integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("depths must not contain duplicates")
    return normalized


def is_hit_at(rank: int | None, depth: int) -> bool:
    """Decide whether an optional 1-based Gold Rank falls within the given positive depth."""

    if type(depth) is not int or depth <= 0:
        raise ValueError("depth must be a positive integer")
    if rank is not None and (type(rank) is not int or rank <= 0):
        raise ValueError("rank must be a positive integer or None")
    return rank is not None and rank <= depth


def gold_chunk_rank(
    ranked_chunk_ids: Sequence[str],
    *,
    complete_gold_chunk_ids: Sequence[str],
    required_gold_chunk_ids: Sequence[str],
) -> int | None:
    """Return the earliest 1-based depth that satisfies the Gold contract.

        If a complete Gold Chunk exists, a hit on any complete Chunk suffices; if the evidence is
        fragmented, the coverage depth is the largest rank at which all required Chunks are present.
    """

    positions: dict[str, int] = {}
    for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        positions.setdefault(chunk_id, rank)

    complete_ids = set(complete_gold_chunk_ids)
    if complete_ids:
        ranks = [positions[chunk_id] for chunk_id in complete_ids if chunk_id in positions]
        return min(ranks) if ranks else None

    required_ids = set(required_gold_chunk_ids)
    if not required_ids or not required_ids <= positions.keys():
        return None
    return max(positions[chunk_id] for chunk_id in required_ids)


def gold_group_rank(
    ranked_chunk_ids: Sequence[str],
    *,
    required_gold_groups: Sequence[Sequence[str]],
) -> int | None:
    """Return the earliest 1-based depth that covers every Gold source group.

        Each inner group holds interchangeable relevant Chunks from the same source, so a hit on
        any one of them counts; the outer groups are all required evidence, so a multi-document
        question must cover every source group. A single-document question uses one group, which
        is equivalent to taking the best rank among all legal Gold Chunks of that source.
    """

    groups = [tuple(group) for group in required_gold_groups]
    if not groups or any(not group for group in groups):
        raise ValueError("required_gold_groups must contain non-empty groups")

    positions: dict[str, int] = {}
    for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        positions.setdefault(chunk_id, rank)

    group_ranks: list[int] = []
    for group in groups:
        # The same source may yield several legal candidates because of Chunk overlap; only the
        # earliest hit is taken.
        matched_ranks = [positions[chunk_id] for chunk_id in set(group) if chunk_id in positions]
        if not matched_ranks:
            return None
        group_ranks.append(min(matched_ranks))
    # The depth at which all sources are first covered at the same time is set by the source
    # recovered last.
    return max(group_ranks)


def gold_chunk_hit(
    ranked_chunk_ids: Sequence[str],
    *,
    depth: int,
    complete_gold_chunk_ids: Sequence[str],
    required_gold_chunk_ids: Sequence[str],
) -> bool:
    """Decide whether the given Candidate depth satisfies the same Gold Chunk contract."""

    rank = gold_chunk_rank(
        ranked_chunk_ids[:depth],
        complete_gold_chunk_ids=complete_gold_chunk_ids,
        required_gold_chunk_ids=required_gold_chunk_ids,
    )
    return is_hit_at(rank, depth)


def hits_at_depths(
    rank: int | None,
    depths: Sequence[int] = (1, 3, 5, 10, 20),
) -> dict[str, bool]:
    """Project one Gold Rank into a stable set of Hit@K boolean metrics."""

    normalized = _validated_depths(depths)
    return {f"hit_at_{depth}": is_hit_at(rank, depth) for depth in normalized}


def aggregate_hits_at_depths(
    ranks: Sequence[int | None],
    depths: Sequence[int] = (1, 3, 5, 10, 20),
) -> dict[str, int]:
    """Aggregate several Gold Ranks so experiment runners do not each maintain the same Hit@K loop."""

    normalized = _validated_depths(depths)
    return {
        f"hit_at_{depth}": sum(is_hit_at(rank, depth) for rank in ranks)
        for depth in normalized
    }

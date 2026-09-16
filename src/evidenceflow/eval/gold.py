"""提供实验间共享的 Chunk-ID Gold Coverage 与 Rank 计算。"""

from collections.abc import Sequence


def _validated_depths(depths: Sequence[int]) -> tuple[int, ...]:
    """规范化正整数深度，并拒绝重复键造成的指标覆盖。"""

    normalized = tuple(depths)
    if not normalized or any(type(depth) is not int or depth <= 0 for depth in normalized):
        raise ValueError("depths must contain positive integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("depths must not contain duplicates")
    return normalized


def is_hit_at(rank: int | None, depth: int) -> bool:
    """判断可选 1-based Gold Rank 是否进入指定正整数深度。"""

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
    """返回达到 Gold 合同所需的最早 1-based 深度。

    若存在完整 Gold Chunk，任一完整 Chunk 命中即可；若证据碎片化，则以全部
    required Chunk 均出现时的最大排名作为覆盖深度。
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
    """返回覆盖每个 Gold 来源组所需的最早 1-based 深度。

    每个内层组表示同一来源中可互换的相关 Chunk，命中任意一个即可；外层各组
    均为回答所需证据，因此多文档题必须覆盖全部来源组。单文档题使用一个组，
    等价于在该来源的所有合法 Gold Chunk 中取最佳排名。
    """

    groups = [tuple(group) for group in required_gold_groups]
    if not groups or any(not group for group in groups):
        raise ValueError("required_gold_groups must contain non-empty groups")

    positions: dict[str, int] = {}
    for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        positions.setdefault(chunk_id, rank)

    group_ranks: list[int] = []
    for group in groups:
        # 同一来源可能因 Chunk overlap 产生多个合法候选，只取最早命中的一个。
        matched_ranks = [positions[chunk_id] for chunk_id in set(group) if chunk_id in positions]
        if not matched_ranks:
            return None
        group_ranks.append(min(matched_ranks))
    # 所有来源首次同时得到覆盖的深度，由最晚被召回的来源决定。
    return max(group_ranks)


def gold_chunk_hit(
    ranked_chunk_ids: Sequence[str],
    *,
    depth: int,
    complete_gold_chunk_ids: Sequence[str],
    required_gold_chunk_ids: Sequence[str],
) -> bool:
    """判断指定 Candidate 深度是否满足同一 Gold Chunk 合同。"""

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
    """把单个 Gold Rank 投影为一组稳定的 Hit@K 布尔指标。"""

    normalized = _validated_depths(depths)
    return {f"hit_at_{depth}": is_hit_at(rank, depth) for depth in normalized}


def aggregate_hits_at_depths(
    ranks: Sequence[int | None],
    depths: Sequence[int] = (1, 3, 5, 10, 20),
) -> dict[str, int]:
    """聚合多个 Gold Rank，避免实验 runner 各自维护相同 Hit@K 循环。"""

    normalized = _validated_depths(depths)
    return {
        f"hit_at_{depth}": sum(is_hit_at(rank, depth) for rank in ranks)
        for depth in normalized
    }

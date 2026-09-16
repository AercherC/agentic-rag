"""提供预先固定随机种子的逐题/题族配对比较。"""

import math
import random


def paired_comparison(baseline: dict[str, float], variant: dict[str, float], *,
                      families: dict[str, str] | None = None, seed: int = 20260905,
                      samples: int = 10000) -> dict:
    """以题族为采样单位保留相关性，返回探索性 CI 与双侧随机化检验。"""
    if not baseline or baseline.keys() != variant.keys():
        raise ValueError("paired query IDs must match exactly")
    if type(samples) is not int or samples < 100:
        raise ValueError("samples must be at least 100")
    if any(not math.isfinite(v) for v in [*baseline.values(), *variant.values()]):
        raise ValueError("metrics must be finite")
    families = families if families is not None else {qid: qid for qid in baseline}
    if families.keys() != baseline.keys():
        raise ValueError("family mapping must cover exactly the paired queries")
    grouped = {}
    deltas = {qid: variant[qid] - baseline[qid] for qid in sorted(baseline)}
    for qid, delta in deltas.items():
        grouped.setdefault(families[qid], []).append(delta)
    clusters = [grouped[key] for key in sorted(grouped)]
    rng = random.Random(seed)
    draws = []
    extreme = 0
    observed = sum(deltas.values()) / len(deltas)
    for _ in range(samples):
        selected = rng.choices(clusters, k=len(clusters))
        draws.append(sum(sum(c) for c in selected) / sum(len(c) for c in selected))
        # 同一题族整体交换系统标签，不能独立打散中英文平行题。
        randomized = sum(rng.choice((-1, 1)) * sum(c) for c in clusters) / len(deltas)
        extreme += abs(randomized) >= abs(observed) - 1e-12
    draws.sort()
    base = sum(baseline.values()) / len(baseline)
    return {"baseline": base, "variant": sum(variant.values()) / len(variant),
            "absolute_delta": observed, "relative_delta": observed / base if base else None,
            "ci95": [draws[int(samples * .025)], draws[min(samples - 1, int(samples * .975))]],
            "wins": sum(d > 0 for d in deltas.values()), "ties": sum(d == 0 for d in deltas.values()),
            "losses": sum(d < 0 for d in deltas.values()), "per_query_delta": deltas,
            "randomization_p": (extreme + 1) / (samples + 1), "seed": seed, "samples": samples,
            "family_count": len(clusters), "interpretation": "exploratory; not proof of non-inferiority"}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """校正预注册的一组比较；不能事后选择性遗漏检验。"""
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in p_values.values()):
        raise ValueError("p values must be finite probabilities")
    result = {}
    previous = 0.0
    for index, (name, p) in enumerate(sorted(p_values.items(), key=lambda pair: pair[1])):
        previous = max(previous, min(1.0, (len(p_values) - index) * p))
        result[name] = previous
    return result

"""A0：把独立 HNSW 回放与冻结 B0 逐题比较，并记录运行环境。

只读取冻结产物，输出写入 artifacts/stage_a/，不修改 evals/ 下的任何文件。
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT / "experiments"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from run_phase15b1_dense_candidate_depth import (  # noqa: E402
    compare_replay,
    load_frozen_inputs,
)

FROZEN_B0 = ROOT / "evals/evaluation_v3/phase15b_hybrid_candidate_v2/production_reranked_baseline.json"
REPLAY = ROOT / "artifacts/stage_a/A0_hnsw_replay.json"
OUT_DIR = ROOT / "artifacts/stage_a"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    # 与项目 write_json 保持一致：UTF-8、不转义中文、键排序，保证可复现。
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    inputs = load_frozen_inputs()
    expected = _read(FROZEN_B0)
    actual = _read(REPLAY)
    question_ids = sorted(expected["final_run"])
    projection_hash = expected.get("active_relevance_projection_hash")
    result = compare_replay(expected, actual, projection_hash, question_ids)
    # 逐题保留差异前后的 Top5，便于报告引用。
    result["per_query_diff"] = {
        qid: {"frozen": expected["final_run"][qid], "replay": actual["final_run"].get(qid)}
        for qid in result["run_mismatches"]
    }
    result["summary_frozen"] = expected.get("summary_at_5")
    result["summary_replay"] = actual.get("summary_at_5")
    _write(OUT_DIR / "A0_replay_compare.json", result)

    env: dict = {"python": sys.version, "platform": platform.platform()}
    import torch

    env["torch"] = torch.__version__
    env["cuda_available"] = bool(torch.cuda.is_available())
    import chromadb

    env["chromadb"] = chromadb.__version__
    try:
        import faiss

        env["faiss_importable"] = True
        env["faiss"] = getattr(faiss, "__version__", "unknown")
    except ImportError:
        env["faiss_importable"] = False
    _write(OUT_DIR / "A0_env.json", env)
    print(json.dumps({k: result[k] for k in ("parity", "run_mismatches", "summary_mismatches", "identity_matches", "query_set_matches", "compared_question_count")}, ensure_ascii=False))
    return 0 if result["parity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

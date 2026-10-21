"""派生式题集工具（draft.py）与 CLI（scripts/e2e_draft.py）的 Fake 测试。

全部用例不联网：题干生成用 ``--dry-run``，探针用注入的 Fake client；
Chroma / Settings / 源文档加载在集成测试里用 monkeypatch 替换。
"""

import json

import pytest

import scripts.e2e_draft as module
from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.domain.document import Document, DocumentSegment
from evidenceflow.eval.e2e import draft
from evidenceflow.eval.stable_mapping import validate_source

# 三个足以通过“自足候选句”过滤的真实形态句子（60–320 字符、非指代开头、含数字或连续专名）。
SENTENCE_PERCENT = (
    "Additionally, we randomly set 1% of the measurements to dmax = 1.6, "
    "and each prediction takes only 0.0012s on a Nvidia GeForce 1080Ti GPU."
)
SENTENCE_CHAMFER = (
    "SnowflakeNet reports a chamfer distance of 7.21 on the PCN benchmark "
    "under the standard evaluation protocol."
)
SENTENCE_GPU = (
    "The Nvidia GeForce GPU is used for all experiments in this paper "
    "and is compared with the baseline method."
)


# ---------- 规范化与句池 ----------

def test_normalize_source_text_does_only_nfkc():
    # 连字 ﬁ 变成 fi，但大小写与空白保持原样（(d) 才做大小写/空白归一）。
    assert draft.normalize_source_text("e\uFB03cient") == "efficient"
    assert draft.normalize_source_text("WEIRD  spacing\nkept") == "WEIRD  spacing\nkept"


def test_sentence_candidates_filters_and_merges_lines():
    assert draft.sentence_candidates(SENTENCE_PERCENT) == [SENTENCE_PERCENT]

    # 指代词/连接词开头、过短、无数字且无连续专名、过长都必须过滤。
    pronoun = "We propose a method that uses 42 layers and achieves strong results on the benchmark."
    too_short = "We set 1%."
    no_number_no_proper = (
        "the model is trained with a large batch size for many epochs and converges quickly "
        "without any tuning on the development set"
    )
    too_long = "The 42 samples " + "x" * 320
    text = "\n".join([pronoun, too_short, no_number_no_proper, too_long])
    assert draft.sentence_candidates(text) == []

    # 跨行排版的同一句要合并成一行；重复句只保留一次。
    wrapped = (
        "Real-world 3D data are often\nincomplete, causing loss in geometric and\n"
        "semantic information for 3D sensors."
    )
    merged = draft.sentence_candidates(wrapped)
    assert len(merged) == 1 and "\n" not in merged[0] and "often incomplete" in merged[0]
    assert draft.sentence_candidates(f"{SENTENCE_CHAMFER} {SENTENCE_CHAMFER}") == [SENTENCE_CHAMFER]

    # 无数字但有两个连续大写开头的专名（Nvidia GeForce）同样可选。
    assert draft.sentence_candidates(SENTENCE_GPU) == [SENTENCE_GPU]


def test_select_spans_prefers_numbers_and_falls_back_to_entities():
    percent = draft.select_spans("we set 1% of the measurements to dmax")
    assert percent and percent[0].span == "1%" and percent[0].kind == "numeric"

    seconds = draft.select_spans("each prediction takes 0.0012s on a Nvidia GPU")
    assert seconds[0].span == "0.0012s" and seconds[0].kind == "numeric"
    assert seconds[0].span in seconds[0].sentence

    entity_only = draft.select_spans("SnowflakeNet is trained on PCN with chamfer distance")
    assert entity_only and entity_only[0].kind == "entity" and entity_only[0].span == "SnowflakeNet"
    # 每个 span 都必须是句子的字面子串，否则锚点无法自证。
    for candidate in percent + seconds + entity_only:
        assert candidate.span in candidate.sentence


def test_split_sentences_folds_line_end_hyphenation():
    """PDF 印刷断词必须在切句时折叠，否则接地检查永远失败（小样校准缺陷 A）。"""

    merged = draft.split_sentences(
        "The PN2 encoder follows the same archi-\ntecture as the SSG network in [38]."
    )
    assert merged == ["The PN2 encoder follows the same architecture as the SSG network in [38]."]
    assert "archi-" not in merged[0]

    # 折行带缩进同样要折叠；跨行但不带连字符的句子仍按空白合并。
    assert draft.split_sentences("the im-\n    pressive result holds.") == ["the impressive result holds."]
    assert draft.split_sentences("often\nincomplete data.") == ["often incomplete data."]

    # 小数不能被切错或折叠（回归保护）。
    assert draft.split_sentences("It takes 0.0012s and scores 7.21 points.") == [
        "It takes 0.0012s and scores 7.21 points."
    ]


def test_select_spans_skips_stopwords_and_reference_numbers():
    """句首虚词/图注词与方括号引用编号都不是答案（小样校准缺陷 B）。"""

    # `The` 是句首虚词，不是可命名答案；这里回落到下一个真实专名。
    stacked = draft.select_spans(
        "The stacked PN encoder used by FC, Folding, PCN-CD and PCN-EMD includes 2 PN layers."
    )
    assert [span.span for span in stacked] and "The" not in [span.span for span in stacked]

    # `Figure` 是图注词、`2` 是图号，答案应是真正的组件名 MLP。
    figure = draft.select_spans(
        "As shown in the lower right of Figure 2, the seed point cloud and the shape code "
        "are sent to MLP respectively and then concatenated."
    )
    assert [span.span for span in figure] == ["MLP"]

    # `[38]` 是文献引用编号，不能作为答案。
    citation = draft.select_spans("The PN2 encoder follows the same architecture as the SSG network in [38].")
    assert "38" not in [span.span for span in citation]

    # 正常专名仍可派生（回归保护）。
    pointnet = draft.select_spans("PointNet++ is trained on ShapeNet.")
    assert pointnet[0].span == "PointNet++"

    # 全部不可用时返回空列表，由调用方按 no_span 丢弃，而不是造一个命中不了的键。
    assert draft.select_spans("The figure and the table are listed above.") == []


# ---------- answer_keys 派生表 ----------

def test_derive_answer_keys_percent_and_unit_forms():
    percent = draft.derive_answer_keys("1%", "we set 1% of the measurements")
    assert len(percent) == 1
    assert {"1%", "1 percent"} <= set(percent[0])
    # 中文题干时额外给“百分之1”。
    zh = draft.derive_answer_keys("1%", "we set 1% of the measurements", query_language="zh")
    assert "百分之1" in zh[0]

    seconds = draft.derive_answer_keys("0.0012s", "it takes 0.0012s on the GPU")
    assert {"0.0012s", "0.0012", "0.0012 s"} <= set(seconds[0])
    assert "1.5" in draft.derive_answer_keys("1.5mm", "the error is 1.5mm")[0]


def test_derive_answer_keys_drops_long_spans_and_short_aliases():
    # 41 字符 ⇒ key_matchable=false，返回空列表。
    assert draft.derive_answer_keys("a" * 41, "x") == []
    # 每个别名规范化后长度 >= 2（裸 "1" 被丢弃，是已知限制）。
    for keys in (
        draft.derive_answer_keys("1%", "we set 1% of it", query_language="zh"),
        draft.derive_answer_keys("0.0012s", "it takes 0.0012s"),
        draft.derive_answer_keys("view-points", "the view-points are listed"),
    ):
        assert keys and all(len(draft.normalize_text(alias)) >= 2 for alias in keys[0])

    # 含连字符的 span 同时给折叠形与空格形。
    hyphen = draft.derive_answer_keys("view-points", "the view-points are listed")
    assert {"viewpoints", "view points"} <= set(hyphen[0])

    # (c)①：至少一个别名来自源句本身。
    sentence = "we set 1% of the measurements"
    keys = draft.derive_answer_keys("1%", sentence)
    assert any(alias in draft.normalize_text(sentence) for alias in keys[0])


# ---------- 3-gram 与连续词长 ----------

def test_three_gram_jaccard_and_longest_common_word_run():
    sentence = "the chamfer distance is 7.21 on pcn"
    assert draft.three_gram_jaccard(sentence, sentence) == 1.0
    assert draft.three_gram_jaccard(sentence, "completely unrelated words here") == 0.0
    # 两侧都为空返回 0.0。
    assert draft.three_gram_jaccard("", "") == 0.0

    nine = "one two three four five six seven eight nine"
    assert draft.longest_common_word_run(nine, nine) == 9
    # 只共享 5 个连续词：不触发 §13.1(c′)④（阈值 > 8）。
    assert draft.longest_common_word_run(
        "alpha beta gamma delta epsilon zeta", "xx alpha beta gamma delta epsilon yy"
    ) == 5
    assert draft.longest_common_word_run(nine, "xx one two three four five six seven eight nine yy") == 9
    assert draft.longest_common_word_run("", "anything") == 0


# ---------- 配额 ----------

def _quota_case(document_id="doc-1", question_type="method", language="en", tags=None):
    return {
        "document_id": document_id,
        "question_type": question_type,
        "query_language": language,
        "tags": tags or [],
    }


def test_plan_quotas_reports_violations_without_raising():
    cases = [_quota_case(tags=["easy"])]
    report = draft.plan_quotas(cases)
    assert report.ok is False
    # 单篇/单题型/难度三档都不足；中英平衡未触发（|1-0| <= 2）。
    assert any("document doc-1" in violation for violation in report.violations)
    assert any("question_type method" in violation for violation in report.violations)
    assert any("difficulty medium" in violation for violation in report.violations)
    assert report.counts["total"] == 1 and report.counts["difficulty"]["easy"] == 1

    # 完全没有难度标签时只报一条“缺标签”，不伪造三档比例。
    no_labels = draft.plan_quotas([_quota_case()])
    assert sum("difficulty labels missing" in violation for violation in no_labels.violations) == 1


# ---------- 分层抽样 ----------

def _candidate(document_id: str, index: int, sentence: str, language: str = "en") -> dict:
    """构造一条 pool 形态的候选句行。"""

    return {
        "candidate_id": f"{document_id}#{index}",
        "chunk_id": f"chunk-{document_id}-{index}",
        "document_id": document_id,
        "sentence": sentence,
        "page_start": 1,
        "page_end": 1,
        "query_language": language,
    }


# 两个可出题题型：数值句（quantitative_results）与专名句（method）。
_NUMERIC_SENTENCE = "The model reduces the chamfer distance to 7.21 on the benchmark."
_ENTITY_SENTENCE = "SnowflakeNet uses a Skip-Transformer decoder for point generation."


def test_stratified_sample_spreads_across_documents():
    # 3 篇论文 × 2 类题型 × 10 条：前缀截断只会取第一篇，分层抽样应三篇都取到。
    candidates = [
        _candidate(document, index, _NUMERIC_SENTENCE if index % 2 == 0 else _ENTITY_SENTENCE)
        for document in ("doc-a", "doc-b", "doc-c")
        for index in range(10)
    ]
    sample = draft.stratified_sample(candidates, limit=12, seed=1)
    assert len(sample) == 12
    assert len({row["document_id"] for row in sample}) == 3
    assert len({row["candidate_id"] for row in sample}) == 12


def test_stratified_sample_is_deterministic_and_seed_sensitive():
    candidates = [_candidate(f"doc-{index % 4}", index, _NUMERIC_SENTENCE) for index in range(40)]
    first = [row["candidate_id"] for row in draft.stratified_sample(candidates, limit=15, seed=1)]
    again = [row["candidate_id"] for row in draft.stratified_sample(candidates, limit=15, seed=1)]
    other = [row["candidate_id"] for row in draft.stratified_sample(candidates, limit=15, seed=2)]
    assert first == again          # 同一 seed 逐元素相同（可复算）
    assert first != other          # 换 seed 得到独立第二遍样本


def test_stratified_sample_limits_and_no_span_bucket():
    candidates = [_candidate("doc-a", index, _NUMERIC_SENTENCE) for index in range(5)]
    # limit 大于总数时返回全部、不重复。
    assert len(draft.stratified_sample(candidates, limit=99, seed=1)) == 5
    assert draft.stratified_sample(candidates, limit=0, seed=1) == []
    with pytest.raises(ValueError):
        draft.stratified_sample(candidates, limit=-1, seed=1)

    # 取不到 span 的句子归入 no_span 层单独轮转，不挤占可出题的层。
    mixed = candidates + [_candidate("doc-b", 99, "The figure and the table are listed above.")]
    sample = draft.stratified_sample(mixed, limit=6, seed=1)
    no_span = [row for row in sample if row["candidate_id"] == "doc-b#99"]
    assert len(no_span) <= 1


# ---------- 锚点定位 ----------

def make_document(pages: dict[int, str], *, document_id="doc-1", content_hash="hash-1") -> Document:
    """构造带页码 Segment 的小文档。"""

    return Document(
        document_id=document_id,
        source_path="paper.pdf",
        file_name="paper.pdf",
        file_type="pdf",
        content_hash=content_hash,
        segments=[DocumentSegment(content=text, page_number=page) for page, text in sorted(pages.items())],
    )


def test_build_evidence_alternative_matches_validate_source():
    sentence = SENTENCE_CHAMFER
    document = make_document({1: f"1. Introduction\n{sentence}\nWe then describe the method."})
    anchor = draft.build_evidence_alternative(document, sentence=sentence, page_start=1, page_end=1)
    assert anchor.document_id == document.document_id
    assert anchor.content_version == document.content_hash
    assert anchor.locator.page_start == anchor.locator.page_end == 1
    assert anchor.locator.occurrence == 1
    assert anchor.locator.block_index is None and anchor.locator.span_start is None
    # 偏移量必须与 stable_mapping.validate_source 的返回一致（同一规范化算法）。
    validation = validate_source(anchor, document)
    assert validation["source_valid"] is True
    assert (anchor.locator.source_span_start, anchor.locator.source_span_end) == (
        validation["source_span_start"],
        validation["source_span_end"],
    )
    assert anchor.locator.left_context and anchor.locator.right_context

    with pytest.raises(ValueError, match="not found"):
        draft.build_evidence_alternative(
            document, sentence="This sentence is not present in the page.", page_start=1, page_end=1
        )
    # 页码不存在时同样定位不到。
    with pytest.raises(ValueError, match="not found"):
        draft.build_evidence_alternative(document, sentence=sentence, page_start=2, page_end=2)


def test_build_evidence_alternative_reuses_hyphen_folding():
    document = make_document({1: "We discuss the view-\npoints of the paper in detail."})
    anchor = draft.build_evidence_alternative(
        document, sentence="the viewpoints of the paper", page_start=1, page_end=1
    )
    validation = validate_source(anchor, document)
    assert validation["source_valid"] is True
    assert anchor.locator.source_span_start == validation["source_span_start"]


# ---------- CLI：构造 3 条 fixture ----------

def _write_chunk_fixture(path):
    rows = [
        {"chunk_id": "k1", "document_id": "d1", "text": SENTENCE_PERCENT, "page_start": 1, "page_end": 1},
        {"chunk_id": "k2", "document_id": "d2", "text": SENTENCE_CHAMFER, "page_start": 2, "page_end": 2},
        {"chunk_id": "k3", "document_id": "d3", "text": SENTENCE_GPU, "page_start": 3, "page_end": 3},
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_pool_from_json_reports_read_only_statistics(tmp_path):
    fixture = _write_chunk_fixture(tmp_path / "chunks.json")
    report_path = tmp_path / "pool.json"
    assert module.main(["pool", "--from-json", str(fixture), "--report", str(report_path)]) == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "b3-pool-v1"
    stats = payload["stats"]
    assert stats["chunks"] == 3 and stats["sentences"] == 3
    assert stats["candidate_sentences"] == 3
    # 只有第一句同时含数字与单位（1% / 0.0012s）。
    assert stats["numeric_unit_sentences"] == 1
    assert set(stats["per_document"]) == {"d1", "d2", "d3"}
    assert all(entry["chunks"] == 1 for entry in stats["per_document"].values())
    assert len(payload["candidates"]) == 3
    assert {"candidate_id", "chunk_id", "document_id", "sentence", "page_start", "page_end"} <= set(
        payload["candidates"][0]
    )

    # --limit 只截断 chunk 数。
    limited = tmp_path / "pool_limited.json"
    assert module.main(["pool", "--from-json", str(fixture), "--report", str(limited), "--limit", "1"]) == 0
    assert json.loads(limited.read_text(encoding="utf-8"))["stats"]["chunks"] == 1


def test_draft_dry_run_writes_expected_jsonl_fields(tmp_path):
    fixture = _write_chunk_fixture(tmp_path / "chunks.json")
    items_path = tmp_path / "items.jsonl"
    assert module.main(["draft", "--from-json", str(fixture), "--out", str(items_path), "--dry-run"]) == 0
    lines = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 3 and all(row["decision"] == "accepted" for row in lines)
    required = {
        "candidate_id", "document_id", "chunk_id", "sentence", "span", "answer_keys", "query",
        "query_language", "question_type", "source", "generation", "decision",
    }
    for row in lines:
        assert required <= set(row)
        assert row["query"] == "<DRY-RUN>" and row["dry_run"] is True
        assert row["generation"]["prompt_version"] == "draft-v1"
        assert row["generation"]["usage"] is None
        assert row["source"] in {"derived_t1", "derived_t2"}
    # 中英按候选序号交替，天然接近各半。
    assert {row["query_language"] for row in lines} == {"en", "zh"}


# ---------- CLI：输出保护与 gates 前置拒绝 ----------

def test_cli_refuses_outputs_under_evals(tmp_path, capsys):
    target_dir = module.ROOT / "evals" / "_b3a2_test_never_created"
    missing = tmp_path / "missing.json"
    commands = [
        ["pool", "--from-json", str(missing), "--report", str(target_dir / "pool.json")],
        ["draft", "--from-json", str(missing), "--out", str(target_dir / "items.jsonl")],
        ["gates", "--items", str(missing), "--out", str(target_dir / "gate.json")],
        ["report", "--gate", str(missing), "--out", str(target_dir / "gate.md")],
    ]
    for argv in commands:
        assert module.main(argv) == 2, argv
    captured = capsys.readouterr()
    assert captured.err.count("REFUSED:") == len(commands)
    # 输出保护在读取输入之前生效：目标目录/文件一个都不应被创建。
    assert not target_dir.exists()


def test_gates_refuses_same_vendor_before_reading_items(tmp_path, monkeypatch, capsys):
    # judge 模型以 deepseek 开头 ⇒ 同厂商；必须在读 items / 连 Chroma 之前拒绝。
    monkeypatch.setenv("EVAL_JUDGE_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "deepseek-chat")
    monkeypatch.setenv("EVAL_JUDGE_API_KEY", "test-key-not-real")
    out = tmp_path / "gate.json"
    assert module.main(["gates", "--items", str(tmp_path / "missing.jsonl"), "--out", str(out)]) == 2
    captured = capsys.readouterr()
    assert "REFUSED:" in captured.err and "vendor" in captured.err
    assert not out.exists()


def test_report_renders_markdown(tmp_path):
    gate = {
        "gates_version": "b3-gates-v1",
        "collection": "col",
        "items": {"total": 1, "accepted": 1, "discarded": 0},
        "gates": {
            name: {"passed": 1, "failed": 0, "failures": []} for name in ("G3", "G4", "G5", "G6")
        },
        "quota": {
            "ok": False,
            "violations": ["document doc-1: 1 < 6"],
            "counts": {"total": 1, "documents": {"doc-1": 1}, "question_types": {"method": 1},
                       "languages": {"en": 1}, "difficulty": {"easy": 1}},
        },
        "discarded": [{"candidate_id": "k9#0", "stage": "draft", "reason": "no_span"}],
        "probe_config": {"judge_host": "open.bigmodel.cn", "judge_model": "glm-4-flash",
                         "generator_host": "api.deepseek.com", "generator_model": "deepseek-flash",
                         "same_vendor": False, "prompt_version": "probe-v1"},
        "rerun_consistency": {"available": False, "checked": 0, "consistent": 0,
                              "value": None, "note": "single-pass items"},
    }
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    out = tmp_path / "gate.md"
    assert module.main(["report", "--gate", str(gate_path), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "## 闸门通过率" in text and "| G3 | 1 | 0 | 100.0% |" in text
    assert "document doc-1: 1 < 6" in text and "k9#0" in text and "probe-v1" in text


# ---------- gates 全链路（Fake client，无网络） ----------

class _FakeProbeClient:
    """上下文含 1% 时答 answered，否则 insufficient_context；不联网。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.last_usage = None
        self.calls = 0

    def generate_json(self, messages, *, schema_name, json_schema):
        self.calls += 1
        user = messages[1].content
        if "1%" in user:
            return json.dumps({"verdict": "answered", "answer": "1%", "reason": "context states 1%"})
        return json.dumps({"verdict": "insufficient_context", "answer": "", "reason": "no answer"})


def _prepare_gates_env(tmp_path, monkeypatch):
    """搭好 gates 的 Fake 环境：假探针、假 Chroma、单页文档；返回 (items_path, dense_path, query)。"""

    monkeypatch.setenv("EVAL_JUDGE_BASE_URL", "https://open.bigmodel.cn")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "glm-4-flash")
    monkeypatch.setenv("EVAL_JUDGE_API_KEY", "test-key-not-real")

    page = f"{SENTENCE_PERCENT} The rest of the page is unrelated."
    document = make_document({1: page}, document_id="d1", content_hash="hash-1")
    gold_chunk = DocumentChunk(chunk_id="k1", document_id="d1", content_hash="hash-1",
                               file_name="paper.pdf", content=page, token_count=50, chunk_index=0,
                               page_start=1, page_end=1)
    other_chunk = DocumentChunk(chunk_id="k2", document_id="d1", content_hash="hash-1",
                                file_name="paper.pdf", content="A completely unrelated baseline paragraph.",
                                token_count=10, chunk_index=1, page_start=1, page_end=1)

    from evidenceflow.config import Settings

    tmp_chroma = tmp_path / "chroma"
    tmp_chroma.mkdir()
    monkeypatch.setattr(module, "Settings", lambda: Settings(
        _env_file=None, vector_db_path=str(tmp_chroma), source_docs_path=str(tmp_path / "docs"),
        llm_base_url="https://api.deepseek.com", llm_model="deepseek-flash",
    ))
    monkeypatch.setattr(module, "resolve_active_collection", lambda settings: "col")
    monkeypatch.setattr(module, "load_source_documents", lambda root: {document.document_id: document})
    monkeypatch.setattr(module, "OpenAICompatibleLLMClient", _FakeProbeClient)

    class FakeStore:
        def __init__(self, path, name, *, create_if_missing=True):
            assert create_if_missing is False

        def iter_all_chunks(self):
            yield gold_chunk
            yield other_chunk

    monkeypatch.setattr(module, "ChromaVectorStore", FakeStore)

    query = "What percentage of the measurements is randomly set to dmax?"
    item = {
        "candidate_id": "k1#0", "document_id": "d1", "chunk_id": "k1",
        "sentence": SENTENCE_PERCENT, "page_start": 1, "page_end": 1, "span": "1%",
        "answer_keys": [["1%", "1 percent"]], "query": query, "query_language": "en",
        "question_type": "quantitative_results", "source": "derived_t2",
        "generation": {"model": "deepseek-flash", "base_url_host": "api.deepseek.com",
                       "prompt_version": "draft-v1", "usage": None},
        "decision": "accepted", "decision_reason": None, "difficulty": None, "dry_run": True,
    }
    items_path = tmp_path / "items.jsonl"
    items_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    dense_path = tmp_path / "dense.json"
    dense_path.write_text(json.dumps({query: ["k2", "k1"]}), encoding="utf-8")
    return items_path, dense_path, query


def test_gates_pipeline_with_fake_probe(tmp_path, monkeypatch):
    items_path, dense_path, _ = _prepare_gates_env(tmp_path, monkeypatch)
    out = tmp_path / "gate.json"
    code = module.main(["gates", "--items", str(items_path), "--out", str(out), "--dense", str(dense_path)])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert {name: gate["failed"] for name, gate in report["gates"].items()} == {
        "G3": 0, "G4": 0, "G5": 0, "G6": 0
    }
    assert report["items"] == {"total": 1, "accepted": 1, "discarded": 0}
    assert report["probe_config"]["same_vendor"] is False
    assert report["quota"]["ok"] is False and "difficulty" in report["quota"]["counts"]
    modes = {entry["mode"] for entry in report["probe_results"]}
    assert modes == {"gold", "distractor"}
    # 未提供 --rerun 时不得伪造一致率：available=false 且 value=null。
    assert report["rerun_consistency"]["available"] is False
    assert report["rerun_consistency"]["value"] is None


# ---------- 题集草案落盘 ----------

def test_gates_dataset_out_writes_valid_dataset(tmp_path, monkeypatch):
    from evidenceflow.eval.e2e import dataset as dataset_module

    items_path, dense_path, query = _prepare_gates_env(tmp_path, monkeypatch)
    out = tmp_path / "gate.json"
    dataset_path = tmp_path / "dataset.json"
    code = module.main([
        "gates", "--items", str(items_path), "--out", str(out), "--dense", str(dense_path),
        "--dataset-out", str(dataset_path),
        "--dataset-id", "rag_e2e_v1",
        "--corpus-version", "test-corpus-version",
    ])
    assert code == 0 and dataset_path.is_file()

    # fingerprint 必须能自校验，否则题集与 mapping 的身份绑定就不可信。
    dataset = dataset_module.load_dataset(dataset_path, verify_fingerprint=True)
    assert dataset.dataset_id == "rag_e2e_v1"
    assert dataset.corpus_version == "test-corpus-version"
    assert len(dataset.cases) == 1
    case = dataset.cases[0]
    assert case.case_id == "E2E-M001"
    assert case.query == query
    assert case.expected_status == "answered"
    # approval_basis 指向报告名 + 该报告字节的 sha256（报告先落盘再算哈希）。
    assert case.provenance.approval_basis.startswith("gate.json#")
    # 落盘固定 LF：dataset.sha256 的口径依赖这一点。
    assert b"\r\n" not in dataset_path.read_bytes()


def test_gates_dataset_out_requires_dataset_id_and_corpus_version(tmp_path, monkeypatch, capsys):
    items_path, dense_path, _ = _prepare_gates_env(tmp_path, monkeypatch)
    out = tmp_path / "gate.json"
    dataset_path = tmp_path / "dataset.json"
    # 只给 --dataset-out 不够：corpus_version 口径未定案，不能由脚本猜默认值。
    code = module.main([
        "gates", "--items", str(items_path), "--out", str(out), "--dense", str(dense_path),
        "--dataset-out", str(dataset_path),
    ])
    assert code == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not dataset_path.exists()


def test_gates_dataset_out_refuses_evals_path(tmp_path, monkeypatch, capsys):
    items_path, dense_path, _ = _prepare_gates_env(tmp_path, monkeypatch)
    out = tmp_path / "gate.json"
    target = module.EVALS_ROOT / "_b3_never_created" / "dataset.json"
    code = module.main([
        "gates", "--items", str(items_path), "--out", str(out), "--dense", str(dense_path),
        "--dataset-out", str(target), "--dataset-id", "x", "--corpus-version", "y",
    ])
    assert code == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not target.exists()


# ---------- 复跑一致性 ----------

def _rerun_item(candidate_id: str, span: str, keys: list[str], *, accepted: bool = True) -> dict:
    """构造一条 draft 形态的候选，供复跑一致性比较。"""

    return {
        "candidate_id": candidate_id,
        "document_id": "d1",
        "sentence": SENTENCE_PERCENT,
        "span": span,
        "answer_keys": [keys],
        "decision": "accepted" if accepted else "discarded",
    }


def test_rerun_consistency_reports_full_agreement():
    first = [_rerun_item("k1#0", "1%", ["1%", "1 percent"]), _rerun_item("k1#1", "1.6", ["1.6"])]
    report, conflicts = module.compute_rerun_consistency(first, first)
    assert report["available"] is True
    assert report["checked"] == 2 and report["consistent"] == 2
    assert report["value"] == 1.0
    assert conflicts == []


def test_rerun_consistency_counts_conflicts_and_accepts_key_overlap():
    first = [
        _rerun_item("k1#0", "1%", ["1%", "1 percent"]),
        _rerun_item("k1#1", "1.6", ["1.6"]),
    ]
    second = [
        # span 相同 → 一致，即使别名集合不完全相同。
        _rerun_item("k1#0", "1%", ["1 percent"]),
        # span 不同且别名无交集 → 冲突。
        _rerun_item("k1#1", "9.9", ["9.9"]),
    ]
    report, conflicts = module.compute_rerun_consistency(first, second)
    assert report["checked"] == 2 and report["consistent"] == 1
    assert report["value"] == 0.5
    assert conflicts == [
        {"candidate_id": "k1#1", "stage": "rerun", "reason": "rerun_conflict"}
    ]


def test_rerun_consistency_ignores_single_sided_acceptance():
    first = [_rerun_item("k1#0", "1%", ["1%"]), _rerun_item("k1#1", "1.6", ["1.6"])]
    # 第二遍把 k1#1 丢了：它没有可比性，既不计入 checked 也不算冲突。
    second = [_rerun_item("k1#0", "1%", ["1%"]), _rerun_item("k1#1", "1.6", ["1.6"], accepted=False)]
    report, conflicts = module.compute_rerun_consistency(first, second)
    assert report["checked"] == 1 and report["consistent"] == 1
    assert conflicts == []


def test_rerun_consistency_empty_denominator_is_null():
    report, conflicts = module.compute_rerun_consistency(
        [_rerun_item("k1#0", "1%", ["1%"], accepted=False)], []
    )
    assert report["available"] is True
    assert report["checked"] == 0
    # 空分母返回 null，而不是伪造 0（§13.1(e) 口径）。
    assert report["value"] is None
    assert conflicts == []

# -*- coding: utf-8 -*-
"""生成 EvidenceFlow 完整流程图（Eval 标准 → RAG 流程 → 受控实验闭环）。"""
from __future__ import annotations
import html
from pathlib import Path

W, H = 980, 1260

def esc(s: str) -> str:
    return html.escape(s, quote=True)

def box(x, y, w, h, title, lines, fill, stroke, title_color="#0f172a", tsize=13.5, lsize=11):
    parts = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>']
    ty = y + 21
    parts.append(f'<text x="{x+w/2:.0f}" y="{ty}" text-anchor="middle" font-size="{tsize}" font-weight="700" fill="{title_color}">{esc(title)}</text>')
    ly = ty + 17
    for ln in lines:
        parts.append(f'<text x="{x+w/2:.0f}" y="{ly}" text-anchor="middle" font-size="{lsize}" fill="#334155">{esc(ln)}</text>')
        ly += 14.5
    return "".join(parts)

def varrow(x, y1, y2, color="#475569"):
    return f'<line x1="{x}" y1="{y1}" y2="{y2}" stroke="{color}" stroke-width="1.6" marker-end="url(#ah)"/>'

def harrow(x1, x2, y, color="#475569"):
    return f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="1.6" marker-end="url(#ah)"/>'

def elbow(x1, y1, x2, y2, color="#475569", label=None):
    midx = (x1 + x2) / 2
    p = (f'<path d="M{x1} {y1} C {x1} {y1+70}, {x2} {y2-70}, {x2} {y2}" '
         f'fill="none" stroke="{color}" stroke-width="1.6" marker-end="url(#ah)"/>')
    if label:
        p += f'<text x="{midx:.0f}" y="{(y1+y2)/2+4:.0f}" text-anchor="middle" font-size="10.5" fill="{color}" font-weight="600">{esc(label)}</text>'
    return p

def container(x, y, w, h, fill, stroke, title, tcolor):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            f'<text x="{x+16}" y="{y+26}" font-size="15" font-weight="800" fill="{tcolor}">{esc(title)}</text>')

svg = []
svg.append(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,PingFang SC,Microsoft YaHei,sans-serif">')
svg.append('<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#475569"/></marker></defs>')

# ---------- Band 1: EVAL STANDARD ----------
svg.append(container(24, 20, 932, 196, "#fffbeb", "#f59e0b", "① EVAL 评测标准（冻结 · 任何代码改动必须过此关）", "#854F0B"))
svg.append('<text x="40" y="62" font-size="11.5" fill="#92400e">与 Runtime 解耦：实验只读冻结 40Q Gold，不修改生产代码；指标/归因/gate 全部确定性可复现</text>')
bx = [40, 218, 396, 574, 752]
bw = 168
for x in bx:
    pass
eval_boxes = [
    ("40题 Gold 数据集", ["document_id + 页码范围", "evidence_anchor 原文锚点", "fine-grained holdout"], "#fef3c7", "#d97706"),
    ("冻结校验", ["dataset_hash 比对", "corpus_hash 比对", "防数据漂移"], "#fef3c7", "#d97706"),
    ("Gold 匹配", ["gold_group_rank", "页码交集 / anchor", "无 LLM Judge"], "#fef3c7", "#d97706"),
    ("评测指标", ["Hit@1/3/5/10/20", "MRR", "nDCG@20"], "#fef3c7", "#d97706"),
    ("Gate 判定", ["失败归因", "candidate/ranking/", "chunk/extract", "runtime_fingerprint"], "#fef3c7", "#d97706"),
]
for x, (t, ls, f, s) in zip(bx, eval_boxes):
    svg.append(box(x, 74, bw, 122, t, ls, f, s))
for i in range(4):
    svg.append(harrow(bx[i]+bw, bx[i+1], 135, "#b45309"))

# ---------- Band 2: RAG PIPELINE ----------
svg.append(container(24, 244, 932, 430, "#f0f9ff", "#3b82f6", "② RAG 处理流程（Runtime 生产路径）", "#185FA5"))
svg.append('<text x="40" y="286" font-size="11.5" fill="#1e40af">双路融合在 Composition Root 装配；Final Top-K=5 固定；相同 document_id / content_hash / chunk_id 数据合同贯穿全链路</text>')

# left column: ingestion (offline)
lx, lw = 56, 352
left = [
    ("本地文档 PDF / MD / TXT", ["data/source_docs/{en,zh}"]),
    ("Loader", ["PyMuPDF 按页提取", "保留 1-based 页码"]),
    ("Cleaner", ["NFC / 空白归一", "MD 保护规则"]),
    ("Chunker", ["BGE-M3 tokenizer", "C512 / O64 滑窗"]),
    ("bge-m3 Embedding", ["Dense 1024 维", "显式 L2 归一化"]),
    ("Chroma Vector Store", ["cosine 距离", "361 chunks (研究集)"]),
]
ly0 = 300
for i, (t, ls) in enumerate(left):
    y = ly0 + i * 56
    svg.append(box(lx, y, lw, 46, t, ls, "#dbeafe", "#2563eb"))
    if i < len(left) - 1:
        svg.append(varrow(lx + lw/2, y + 46, y + 56, "#2563eb"))

# right column: online QA
rx, rw = 572, 352
right = [
    ("用户 Query", ["shell 引号保护"]),
    ("双路 Dense", ["Original + 翻译→Dense", "Candidate Top-10 ×2"]),
    ("Equal RRF 融合", ["k=60, w=1:1", "chunk_id 去重"]),
    ("Final Top-5", ["融合分数排序"]),
    ("ContextBuilder", ["4096 tok 预算", "[S1] [S2] … 标签"]),
    ("Generator + Citation", ["temp0 + STATUS 协议", "[Sx] allowlist → RAGAnswer"]),
]
for i, (t, ls) in enumerate(right):
    y = ly0 + i * 56
    svg.append(box(rx, y, rw, 46, t, ls, "#e0f2fe", "#0284c7"))
    if i < len(right) - 1:
        svg.append(varrow(rx + rw/2, y + 46, y + 56, "#0284c7"))

# retrieval elbow: Chroma (L6) -> 双路 Dense (R2)
l6 = left[5]
r2 = right[1]
svg.append(elbow(lx + lw, ly0 + 5*56 + 23, rx, ly0 + 1*56 + 23, "#0f766e", "向量检索读取"))

# ---------- Band 3: CONTROLLED EXPERIMENT LOOP ----------
svg.append(container(24, 700, 932, 524, "#ecfdf5", "#059669", "③ 受控实验闭环（改动验证方法 · 离线 · 不改 Runtime）", "#0F6E56"))
svg.append('<text x="40" y="742" font-size="11.5" fill="#065f46">Gate 不通过即停止；实验产物带 runtime_fingerprint（关键源文件 SHA-256），绝不回写 Runtime / PROJECT_SPEC</text>')

# cycle nodes
cyc = [
    ("改动提案 / 假设", ["明确干预与对照"]),
    ("冻结配置", ["runtime_fingerprint", "记录源文件哈希"]),
    ("40Q 上跑", ["离线·不改 Runtime", "add-only / 受控"]),
    ("计算指标", ["对比基线", "Hit/MRR/nDCG"]),
    ("Gate 判定", ["pass / fail", "归因 + 结论"]),
]
cw, ch = 150, 66
cx = [40, 216, 392, 568, 744]
cyy = 760
for x, (t, ls) in zip(cx, cyc):
    svg.append(box(x, cyy, cw, ch, t, ls, "#d1fae5", "#059669"))
for i in range(4):
    svg.append(harrow(cx[i]+cw, cx[i+1], cyy + ch/2, "#047857"))
# return arrow
svg.append(f'<path d="M{cx[4]+cw/2:.0f} {cyy+ch} C {cx[4]+cw/2:.0f} {cyy+ch+78}, {cx[0]+cw/2:.0f} {cyy+ch+78}, {cx[0]+cw/2:.0f} {cyy+ch}" fill="none" stroke="#047857" stroke-width="1.6" marker-end="url(#ah)"/>')
svg.append(f'<text x="{(cx[0]+cx[4]+cw)/2:.0f}" y="{cyy+ch+70}" text-anchor="middle" font-size="11" fill="#047857" font-weight="700">下一轮实验 / 停止（不回写 Runtime）</text>')

# completed experiments table
tx, tw = 40, 900
cols = [("实验", 200), ("配置 / 干预", 210), ("关键结果", 300), ("Gate / 结论", 190)]
colx = [tx]
for _, w in cols[:-1]:
    colx.append(colx[-1] + w)
tyh = 52
hy = 866
# header
svg.append(f'<rect x="{tx}" y="{hy}" width="{tw}" height="30" fill="#065f46"/>')
for (name, w), x in zip(cols, colx):
    svg.append(f'<text x="{x+10}" y="{hy+20}" font-size="12" font-weight="700" fill="#ffffff">{esc(name)}</text>')
rows = [
    ("research_dense_baseline_v1", "Runtime Dense 双路", "Hit@5 .475 · MRR .302 · nDCG@20 .404", "基线", "#64748b"),
    ("candidate_failure_diagnosis", "固定 12 题失败归因", "候选生成失败 12 / 排序失败 7", "—", "#64748b"),
    ("references_aware", "References 过滤", "救回 1 题但误伤 1 题", "PARTIAL", "#b45309"),
    ("candidate_augmentation", "+BM25 Top-10 add-only", "候选覆盖 26 → 31", "PASS · 不解决排序", "#0f766e"),
    ("candidate_reranking", "重排增强池(dense/hybrid)", "未把覆盖转化为 Top-5", "PASS → 不集成", "#0f766e"),
]
ry = hy + 30
for ridx, (name, cfg, res, gate, gc) in enumerate(rows):
    bg = "#f0fdfa" if ridx % 2 == 0 else "#ffffff"
    svg.append(f'<rect x="{tx}" y="{ry}" width="{tw}" height="{tyh}" fill="{bg}" stroke="#cbd5e1" stroke-width="1"/>')
    cells = [name, cfg, res, gate]
    for ci, (cell, x, (_n, w)) in enumerate(zip(cells, colx, cols)):
        color = gc if ci == 3 else "#1e293b"
        fw = "700" if ci in (0, 3) else "400"
        svg.append(f'<text x="{x+10}" y="{ry+22}" font-size="11.5" font-weight="{fw}" fill="{color}">{esc(cell)}</text>')
    ry += tyh

svg.append(f'<text x="40" y="{ry+34}" font-size="12" fill="#065f46" font-weight="600">结论：评测标准完整且与 Runtime 解耦；任何"提升"都通过该闭环以冻结 40Q 验证，证据不足不增加技术复杂度。</text>')

svg.append('</svg>')

svg_str = "\n".join(svg)

html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EvidenceFlow · 完整流程图（Eval 标准 → RAG 流程 → 受控实验闭环）</title>
<style>
  :root{{--bg:#ffffff;--panel:#f8fafc;--ink:#0f172a;--muted:#475569;--line:#e2e8f0;}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;line-height:1.6;font-size:14px}}
  .wrap{{max-width:1020px;margin:0 auto;padding:28px 22px 56px}}
  h1{{font-size:22px;margin:0 0 4px}}
  .sub{{color:var(--muted);font-size:13px;margin-bottom:18px}}
  .diagram{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:6px;margin:14px 0;overflow-x:auto}}
  .foot{{color:var(--muted);font-size:12px;margin-top:26px;border-top:1px solid var(--line);padding-top:12px}}
</style>
</head>
<body>
<div class="wrap">
  <h1>EvidenceFlow 完整流程图</h1>
  <p class="sub">三大板块：① 冻结 Eval 评测标准（改动必须过此关）· ② RAG 处理流程（生产 Runtime）· ③ 受控实验闭环（改动验证方法）+ 已完成实验清单</p>
  <div class="diagram">
{svg_str}
  </div>
  <div class="foot">对应规范 PROJECT_SPEC.md §18–§20（Gold 标注 / 命中判定 / 指标）与 §17.2（Retrieval 冻结，除非稳定 Evaluation 证明瓶颈）；实验产物均带 runtime_fingerprint。</div>
</div>
</body>
</html>
"""

# 输出位置不随调用者的工作目录改变。
out = Path(__file__).resolve().parents[2] / "docs/evidenceflow_flowchart.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html_doc)
print("written", out, "svg bytes", len(svg_str))

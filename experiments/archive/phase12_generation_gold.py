"""已冻结的 Phase 12 Generation Gold 历史脚本。

编写纪律（对应计划 §7 / §8）：
1. reference_answer 只描述 Gold Chunk 明写的事实，不引入语料外知识；
2. 一条 required_claim = 一个可独立判断的事实；
3. acceptable_evidence 精确到 chunk_id 前缀（prepare 时解析为完整 id），
   不按 document_id 粗粒度判定；
4. answerability 由语料证据决定，不参考任何模型实际输出；
5. INSUFFICIENT_CONTEXT 题必须由零命中的关键术语支撑，且由 prepare 复验。

ARR-* 题目只引用 canonical 40Q 的 query 文本（由 prepare 从数据集读取），
本模块只补充 Generation 侧字段，不修改 Retrieval canonical 40Q。
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# 从 canonical 40Q 派生的 ANSWERABLE 题
# ---------------------------------------------------------------------------
ANSWERABLE_GOLD: dict[str, dict[str, Any]] = {
    "ARR-002": {
        "reference_answer": (
            "PoinTr 指出，真实应用中采集的点云常因三个原因而不完整："
            "传感器分辨率有限（limited sensor resolution）、单一视角（single viewpoint）与遮挡（occlusion）。"
        ),
        "required_claims": [
            {"claim_id": "C1", "text": "传感器分辨率有限会导致点云不完整。",
             "required": True, "evidence": ["55f90c54fa70"]},
            {"claim_id": "C2", "text": "单一视角会导致点云不完整。",
             "required": True, "evidence": ["55f90c54fa70"]},
            {"claim_id": "C3", "text": "遮挡会导致点云不完整。",
             "required": True, "evidence": ["55f90c54fa70"]},
        ],
    },
    "ARR-004": {
        "reference_answer": (
            "SVDFormer 针对的是真实扫描点云的稀疏性（sparsity）与大尺度结构不完整性"
            "（large structural incompleteness）；其双路径设计还能处理高度稀疏的真实世界扫描"
            "（highly sparse real-world scans）。"
        ),
        "required_claims": [
            {"claim_id": "C1", "text": "捕获的点云存在稀疏性问题。",
             "required": True, "evidence": ["d9b0f28a3094"]},
            {"claim_id": "C2", "text": "捕获的点云存在大尺度结构不完整性问题。",
             "required": True, "evidence": ["d9b0f28a3094"]},
            {"claim_id": "C3", "text": "该方法能够处理高度稀疏的真实世界扫描。",
             "required": True, "evidence": ["d9b0f28a3094"]},
        ],
    },
    "ARR-009": {
        "reference_answer": (
            "PoinTr represents a point cloud as a set of unordered groups of points with position "
            "embeddings, converts it into a sequence of point proxies, and uses those proxies as the "
            "inputs of the transformer."
        ),
        "required_claims": [
            {"claim_id": "C1", "text": "The point cloud is represented as a set of unordered groups "
                                       "of points with position embeddings.",
             "required": True, "evidence": ["55f90c54fa70"]},
            {"claim_id": "C2", "text": "The point cloud is converted into a sequence of point proxies.",
             "required": True, "evidence": ["55f90c54fa70", "425d2d1a1501"]},
            {"claim_id": "C3", "text": "The point proxies are used as the inputs of the transformer model.",
             "required": True, "evidence": ["425d2d1a1501"]},
        ],
    },
    "ARR-010": {
        "reference_answer": (
            "SnowflakeNet 的架构中包含 Snowflake Point Deconvolution（SPD，雪花点反卷积）模块；"
            "SPD 中引入 skip-transformer，用于学习最适合局部区域的点分裂模式。"
        ),
        "required_claims": [
            {"claim_id": "C1", "text": "点反卷积模块名为 Snowflake Point Deconvolution（SPD）。",
             "required": True, "evidence": ["49d718c954f9"]},
            {"claim_id": "C2", "text": "SPD 中引入 skip-transformer 来学习点分裂模式。",
             "required": False, "evidence": ["49d718c954f9", "11aa04fab98f"]},
        ],
    },
    "ARR-011": {
        "reference_answer": (
            "SeedFormer introduces a new shape representation called Patch Seeds, which captures general "
            "structures from partial inputs and also preserves regional information of local patterns."
        ),
        "required_claims": [
            {"claim_id": "C1", "text": "The new shape representation is called Patch Seeds.",
             "required": True, "evidence": ["b46ad0c4c4c6"]},
            {"claim_id": "C2", "text": "Patch Seeds capture general structures from partial inputs.",
             "required": True, "evidence": ["b46ad0c4c4c6"]},
            {"claim_id": "C3", "text": "Patch Seeds preserve regional information of local patterns.",
             "required": True, "evidence": ["b46ad0c4c4c6"]},
        ],
    },
    "ARR-012": {
        "reference_answer": (
            "ProxyFormer 将点云划分为已有（输入）与缺失（待预测）两部分，两部分通过各自的 proxy 交换信息；"
            "它设计 missing part sensitive transformer，把随机正态分布转换为合理的位置信息，"
            "并用 proxy alignment 精修缺失 proxy，使预测结果对缺失部分的特征与位置更敏感。"
        ),
        "required_claims": [
            {"claim_id": "C1", "text": "点云被划分为已有（输入）部分与缺失（待预测）部分，"
                                       "两部分通过 proxy 进行信息交互。",
             "required": True, "evidence": ["9119edb32f2d"]},
            {"claim_id": "C2", "text": "设计 missing part sensitive transformer，"
                                       "将随机正态分布转换为合理的位置信息。",
             "required": True, "evidence": ["9119edb32f2d"]},
            {"claim_id": "C3", "text": "使用 proxy alignment 精修缺失 proxy，"
                                       "使预测的 proxy 对缺失部分的特征与位置更敏感。",
             "required": True, "evidence": ["9119edb32f2d"]},
        ],
    },
    "ARR-013": {
        "reference_answer": (
            "The Self-structure Dual-Generator (SDG) consists of a pair of parallel refinement units "
            "called Structure Analysis and Similarity Alignment."
        ),
        "required_claims": [
            {"claim_id": "C1", "text": "The refinement module is called Self-structure Dual-Generator (SDG).",
             "required": True, "evidence": ["2f992241e43d"]},
            {"claim_id": "C2", "text": "SDG contains a pair of parallel refinement units.",
             "required": True, "evidence": ["2f992241e43d"]},
            {"claim_id": "C3", "text": "The two units are named Structure Analysis and Similarity Alignment.",
             "required": True, "evidence": ["2f992241e43d"]},
        ],
    },
    "ARR-014": {
        "reference_answer": (
            "HyperRS（Hyper Refinement Stack）通过一系列超图学习阶段由粗到细地逐步捕获高阶相关性；"
            "A-HGNN（Anchor-based Hypergraph Neural Network）采用两阶段采样策略构建协同超图，"
            "以稳健地建模全局结构。"
        ),
        "required_claims": [
            {"claim_id": "C1", "text": "HyperRS 即 Hyper Refinement Stack。",
             "required": True, "evidence": ["ae8a725ffac5"]},
            {"claim_id": "C2", "text": "HyperRS 通过一系列超图学习阶段，由粗到细逐步捕获高阶相关性。",
             "required": True, "evidence": ["0b3ae8015e61"]},
            {"claim_id": "C3", "text": "A-HGNN 即 Anchor-based Hypergraph Neural Network。",
             "required": True, "evidence": ["ae8a725ffac5", "0b3ae8015e61"]},
            {"claim_id": "C4", "text": "A-HGNN 采用两阶段采样策略构建协同超图，以稳健建模全局结构。",
             "required": True, "evidence": ["0b3ae8015e61"]},
        ],
    },
    "ARR-016": {
        "reference_answer": (
            "SnowflakeNet 主要使用 Chamfer distance（CD）作为损失函数：completion loss 是 "
            "{Pc, P1, P2, P3} 上四个 CD 损失之和，再与 partial matching loss 形式的 preservation loss "
            "组合为 L = Lcompletion + λLpreservation。"
        ),
        "required_claims": [
            {"claim_id": "C1", "text": "主要损失函数是 Chamfer distance（CD）。",
             "required": True, "evidence": ["e4ca9a1eef18"]},
            {"claim_id": "C2", "text": "completion loss 定义为 {Pc, P1, P2, P3} 上四个 CD 损失之和。",
             "required": True, "evidence": ["e4ca9a1eef18"]},
            {"claim_id": "C3", "text": "额外使用 partial matching loss 作为 preservation loss。",
             "required": True, "evidence": ["e4ca9a1eef18"]},
            {"claim_id": "C4", "text": "总损失为 L = Lcompletion + λLpreservation。",
             "required": False, "evidence": ["e4ca9a1eef18"]},
        ],
    },
    "ARR-018": {
        "reference_answer": (
            "在 ShapeNet-55/34 上，ProxyFormer 输入的不完整点云为 2048 个点，完整点云包含 8192 个点。"
        ),
        "required_claims": [
            {"claim_id": "C1", "text": "输入的不完整点云有 2048 个点。",
             "required": True, "evidence": ["b00784ef0d51"]},
            {"claim_id": "C2", "text": "完整点云包含 8192 个点。",
             "required": True, "evidence": ["b00784ef0d51"]},
        ],
    },
    "ARR-019": {
        "reference_answer": (
            "Each ground-truth point cloud contains 16384 points, evenly sampled from the shape surface."
        ),
        "required_claims": [
            {"claim_id": "C1", "text": "Each ground-truth point cloud contains 16384 points.",
             "required": True, "evidence": ["1d7ad9b06566"]},
            {"claim_id": "C2", "text": "The ground-truth points are evenly sampled from the shape surface.",
             "required": True, "evidence": ["1d7ad9b06566"]},
        ],
    },
    "ARR-022": {
        "reference_answer": "SnowflakeNet 的主要实验在两个广泛使用的基准上进行：PCN 与 Completion3D。",
        "required_claims": [
            {"claim_id": "C1", "text": "两个基准之一是 PCN。",
             "required": True, "evidence": ["e4ca9a1eef18"]},
            {"claim_id": "C2", "text": "另一个基准是 Completion3D。",
             "required": True, "evidence": ["e4ca9a1eef18"]},
        ],
    },
    "ARR-029": {
        "reference_answer": (
            "SVDFormer reports ShapeNet-34 results in Table 3 separately for 34 seen categories and 21 "
            "unseen categories with three metrics (CD, DCD and F1), and states it achieves the best "
            "performance in terms of all three metrics."
        ),
        "required_claims": [
            {"claim_id": "C1", "text": "Results are reported separately for 34 seen categories "
                                       "and 21 unseen categories.",
             "required": True, "evidence": ["7deb84071070"]},
            {"claim_id": "C2", "text": "Three metrics are used: CD, DCD and F1.",
             "required": True, "evidence": ["7deb84071070"]},
            {"claim_id": "C3", "text": "SVDFormer achieves the best performance in terms of all three metrics.",
             "required": True, "evidence": ["7deb84071070"]},
        ],
    },
    "ARR-030": {
        "reference_answer": "PointAttN 的 PCN 结果表采用 L1 版本的 Chamfer distance（CD）作为指标。",
        "required_claims": [
            {"claim_id": "C1", "text": "PCN 结果表使用 L1 版本的 Chamfer distance（CD）。",
             "required": True, "evidence": ["a6783f529ec1"]},
            {"claim_id": "C2", "text": "PointAttN 在该表上的 Average 结果为 6.86。",
             "required": False, "evidence": ["a6783f529ec1"]},
        ],
    },
    "ARR-031": {
        "reference_answer": (
            "AdaPoinTr ablates the point-proxy formulation Fi = F'i + φ(pi): removing the positional "
            "embedding drops performance by 0.48, and replacing the mini-DGCNN feature extractor with "
            "PointNet drops it by 0.23."
        ),
        "required_claims": [
            {"claim_id": "C1", "text": "Point proxies are built as Fi = F'i + φ(pi), where F'i is the local "
                                       "structure feature and φ(pi) is the positional embedding.",
             "required": True, "evidence": ["8cc960f0638d"]},
            {"claim_id": "C2", "text": "Abandoning the positional embedding degrades performance by 0.48.",
             "required": True, "evidence": ["8cc960f0638d"]},
            {"claim_id": "C3", "text": "Replacing mini-DGCNN with PointNet as the feature extractor "
                                       "degrades performance by 0.23.",
             "required": True, "evidence": ["8cc960f0638d"]},
        ],
    },
    "ARR-036": {
        "reference_answer": (
            "PointAttN 是一个充分利用于点云补全的注意力机制的深度神经网络；其动机是扩大处理三维点云时"
            "的感受野，以更好地保留几何结构与上下文信息；它通过 cross-attention 与 self-attention "
            "感知局部细节与全局上下文，从而消除了 kNN 之类的显式局部区域划分。"
        ),
        "required_claims": [
            {"claim_id": "C1", "text": "PointAttN 是一个充分利用注意力机制完成点云补全的深度神经网络。",
             "required": True, "evidence": ["77534607e444"]},
            {"claim_id": "C2", "text": "其动机是扩大处理三维点云时的感受野，以更好保留几何结构与上下文信息。",
             "required": True, "evidence": ["77534607e444"]},
            {"claim_id": "C3", "text": "通过 cross-attention 与 self-attention 分别感知局部细节与全局上下文。",
             "required": True, "evidence": ["77534607e444"]},
            {"claim_id": "C4", "text": "该方法消除了 kNN 之类的显式局部区域划分。",
             "required": True, "evidence": ["77534607e444"]},
        ],
    },
    "ARR-039": {
        "reference_answer": (
            "SnowflakeNet uses a skip-transformer inside Snowflake Point Deconvolution (SPD): attention "
            "summarizes the point-splitting patterns used in the previous SPD layer to produce the "
            "splitting in the current layer. PointAttN instead uses self-attention and cross-attention to "
            "capture local details and global context, with a geometric details perception (GDP) block "
            "applying cross-attention between original and down-sampled features, which removes explicit "
            "local region partition such as kNN."
        ),
        "required_claims": [
            {"claim_id": "C1", "text": "SnowflakeNet uses a skip-transformer inside SPD.",
             "required": True, "evidence": ["11aa04fab98f", "4eeb3b90eb7d"]},
            {"claim_id": "C2", "text": "Skip-transformer uses attention to summarize splitting patterns of "
                                       "the previous SPD layer to produce the current splitting.",
             "required": True, "evidence": ["11aa04fab98f"]},
            {"claim_id": "C3", "text": "PointAttN uses self-attention and cross-attention to capture both "
                                       "local and global geometric context.",
             "required": True, "evidence": ["c9943c80def4"]},
            {"claim_id": "C4", "text": "PointAttN's GDP block applies cross-attention between the original "
                                       "point cloud feature and the down-sampled point cloud feature.",
             "required": True, "evidence": ["c9943c80def4"]},
            {"claim_id": "C5", "text": "PointAttN eliminates explicit local region partition such as kNN.",
             "required": True, "evidence": ["c9943c80def4"]},
        ],
    },
    # ---- 以下为 Retrieval 失败题（Phase 9 R@10 = 0），用于分层归因诊断 ----
    "ARR-007": {
        "reference_answer": (
            "The PCN decoder produces two outputs: a coarse output point cloud Ycoarse and a detailed "
            "output point cloud Ydetail, with the coarse output generated first."
        ),
        "required_claims": [
            {"claim_id": "C1", "text": "The decoder produces a coarse output Ycoarse.",
             "required": True, "evidence": ["d3e2a9c34202"]},
            {"claim_id": "C2", "text": "The decoder produces a detailed output Ydetail.",
             "required": True, "evidence": ["d3e2a9c34202"]},
            {"claim_id": "C3", "text": "The encoder abstracts the input X into a feature vector v and the "
                                       "coarse output is generated before the detailed output.",
             "required": True, "evidence": ["d3e2a9c34202"]},
        ],
    },
    "ARR-037": {
        "reference_answer": (
            "PoinTr replaces fixed queries with dynamic queries generated by a query generation module "
            "that summarizes encoder features and represents the initial sketch of the missing points. "
            "AdaPoinTr likewise adopts dynamic query embeddings in the decoder, in contrast to the "
            "learnable static query embedding of prior work, which makes the decoder more flexible and "
            "adjustable for different objects and their missing information."
        ),
        "required_claims": [
            {"claim_id": "C1", "text": "PoinTr uses dynamic queries instead of fixed queries, generated by "
                                       "a query generation module that summarizes encoder features.",
             "required": True, "evidence": ["425d2d1a1501"]},
            {"claim_id": "C2", "text": "PoinTr's dynamic queries represent the initial sketch "
                                       "of the missing points.",
             "required": True, "evidence": ["425d2d1a1501"]},
            {"claim_id": "C3", "text": "AdaPoinTr uses dynamic query embeddings in the decoder, unlike the "
                                       "learnable static query embedding in prior work.",
             "required": True, "evidence": ["a5063069a6d7"]},
            {"claim_id": "C4", "text": "AdaPoinTr's dynamic query scheme makes the decoder more flexible "
                                       "and adjustable for different object types and missing information.",
             "required": True, "evidence": ["a5063069a6d7"]},
        ],
    },
    "ARR-040": {
        "reference_answer": (
            "SVDFormer 通过 Self-structure Dual-Generator 结合学习到的形状先验与几何自相似性建模结构关系："
            "Structure Analysis 单元显式编码粗点云的局部不完整性，Similarity Alignment 单元为每个点寻找"
            "相似结构的特征。Hyper-PCN 则用超图显式建模点云内部复杂的高阶相关性：HyperRS 由粗到细逐步"
            "捕获高阶相关性，A-HGNN 以两阶段采样构建协同超图来稳健建模全局结构。"
        ),
        "required_claims": [
            {"claim_id": "C1", "text": "SVDFormer 引入 Self-structure Dual-Generator，"
                                       "结合学习到的形状先验与几何自相似性。",
             "required": True, "evidence": ["2f992241e43d", "d9b0f28a3094"]},
            {"claim_id": "C2", "text": "Structure Analysis 单元显式编码生成粗点云的局部不完整性。",
             "required": True, "evidence": ["d9b0f28a3094"]},
            {"claim_id": "C3", "text": "Similarity Alignment 单元为每个点寻找相似结构的特征。",
             "required": True, "evidence": ["d9b0f28a3094"]},
            {"claim_id": "C4", "text": "Hyper-PCN 使用超图显式建模点云内部复杂的高阶相关性。",
             "required": True, "evidence": ["0b3ae8015e61"]},
            {"claim_id": "C5", "text": "HyperRS 由粗到细逐步捕获高阶相关性；"
                                       "A-HGNN 以两阶段采样构建协同超图来稳健建模全局结构。",
             "required": True, "evidence": ["0b3ae8015e61"]},
        ],
    },
}

# ---------------------------------------------------------------------------
# 新增的 INSUFFICIENT_CONTEXT 题（不属于 canonical 40Q，独立 ID 空间）
# absence_terms 中的术语必须在本模块 prepare 时对全语料复验为 0 命中。
# ---------------------------------------------------------------------------
INSUFFICIENT_CONTEXT_GOLD: dict[str, dict[str, Any]] = {
    "GEN-U01": {
        "query": "What BLEU score did the original Transformer paper report on WMT 2014 "
                 "English-to-German translation?",
        "query_language": "en",
        "corpus_language": "n/a",
        "question_type": "out_of_corpus_fact",
        "reference_answer": None,
        "required_claims": [],
        "absence_terms": ["BLEU", "WMT"],
        "rationale": "语料为点云补全论文，不含机器翻译评测；BLEU/WMT 全语料零命中。",
    },
    "GEN-U02": {
        "query": "SnowflakeNet 的训练使用了多少块 A100 GPU？",
        "query_language": "zh",
        "corpus_language": "n/a",
        "question_type": "corpus_absent_hardware_fact",
        "reference_answer": None,
        "required_claims": [],
        "absence_terms": ["A100"],
        "rationale": "语料未说明训练所用 GPU 型号；A100 全语料零命中。",
    },
    "GEN-U03": {
        "query": "Which software license is the SeedFormer source code released under?",
        "query_language": "en",
        "corpus_language": "n/a",
        "question_type": "corpus_absent_metadata",
        "reference_answer": None,
        "required_claims": [],
        "absence_terms": ["license"],
        "rationale": "论文只给出代码仓库链接，未声明开源许可证；license 全语料零命中。",
    },
    "GEN-U04": {
        "query": "Point-BERT 的自监督预训练掩码比例（mask ratio）是多少？",
        "query_language": "zh",
        "corpus_language": "n/a",
        "question_type": "out_of_corpus_method",
        "reference_answer": None,
        "required_claims": [],
        "absence_terms": ["Point-BERT", "mask ratio"],
        "rationale": "Point-BERT 不在语料的 10 篇论文中；Point-BERT 与 mask ratio 均零命中。",
    },
    "GEN-U05": {
        "query": "PCN 论文报告的点云法向量估计（normal estimation）准确率是多少？",
        "query_language": "zh",
        "corpus_language": "n/a",
        "question_type": "unsupported_task_fact",
        "reference_answer": None,
        "required_claims": [],
        "absence_terms": ["normal estimation"],
        "rationale": "PCN 论文做的是形状补全，未做法向量估计；normal estimation 零命中。",
    },
}

# 对 INSUFFICIENT_CONTEXT 题，任何给出具体数值/名称的回答都属于幻觉。
INSUFFICIENT_CONTEXT_FORBIDDEN = (
    "给出任何具体数值、型号、许可证名称或比例作为确定答案（而非说明语料未提供）"
)

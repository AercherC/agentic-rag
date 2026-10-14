"""EvidenceFlow 应用配置定义。"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"

# Chunk 策略的合法取值（§9.3）与 fixed 身份 Collection 名单（§9.4 身份隔离守卫）。
CHUNK_STRATEGIES = ("hybrid", "fixed")
FIXED_IDENTITY_COLLECTIONS = frozenset(
    {
        "ef_research_bge_m3_c512_o64_v1",
        "ef_bge_m3_c512_o64",
    }
)


class Settings(BaseSettings):
    """从环境变量和可选的 .env 文件加载统一运行配置。"""

    # 环境变量按大小写不敏感方式匹配，使 LLM_API_KEY 等大写名称
    # 能够映射到下方符合 Python 命名习惯的小写属性。
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm_api_key: str | None = Field(
        default=None,
        description="OpenAI-compatible LLM 的 API Key，仅从运行环境读取。",
    )
    llm_base_url: str = Field(
        default="https://api-inference.modelscope.cn/v1/",
        description="OpenAI-compatible LLM API 的基础地址。",
    )
    llm_model: str = Field(
        default="Qwen/Qwen3.5-35B-A3B",
        description="当前 LLM 服务商提供的模型名称。",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0,
        le=2,
        allow_inf_nan=False,
        description="Grounded Answer 的采样温度；默认零以提高可复现性。",
    )
    llm_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        allow_inf_nan=False,
        description="单次 OpenAI-compatible LLM 请求的超时秒数。",
    )
    llm_max_attempts: int = Field(
        default=2,
        ge=1,
        le=2,
        description="单次 LLM 业务调用的最大尝试次数；仅瞬态错误允许第二次尝试。",
    )
    llm_thinking: Literal["auto", "enabled", "disabled"] = Field(
        default="auto",
        description=(
            "是否向 OpenAI-compatible 服务发送 thinking 开关。auto = 不发送，沿用服务端默认"
            "（DeepSeek 下思考默认开启）；enabled/disabled 显式开或关。"
        ),
    )

    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="文档和查询共同使用的本地 Embedding 模型。",
    )
    source_docs_path: Path = Field(
        default=Path("data/source_docs"),
        description="正式摄取根目录；所有 document_id 均相对此路径稳定生成。",
    )

    chunk_size: int = Field(
        default=512,
        gt=0,
        description="单个 Chunk 允许包含的最大 Token 数，按 Embedding Tokenizer 计算。",
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        description="相邻 Chunk 之间重复保留的 Token 数（仅 fixed 策略使用；hybrid 冻结为 0）。",
    )
    chunk_strategy: Literal["hybrid", "fixed"] = Field(
        default="hybrid",
        description=(
            "Chunk 切分策略（Phase 13F 晋升）：hybrid = 结构解析 + standalone 独立出块 "
            "+ section 聚合（PH1 语义，窗口零重叠）；fixed = 原固定滑动窗口（回滚通道）。"
        ),
    )

    max_context_tokens: int = Field(
        default=4096,
        gt=0,
        description="LLM Context 可容纳完整 Chunk 的最大 Token 预算。",
    )

    reranker_model: str = Field(
        default=DEFAULT_RERANKER_MODEL,
        min_length=1,
        description="正式 Cross-Encoder 模型名称或本地路径。",
    )
    reranker_input_k: int = Field(
        default=20,
        gt=0,
        description="Original Dense ANN 送入 Reranker 的固定候选数量。",
    )
    reranker_final_k: int = Field(
        default=5,
        gt=0,
        description="Learned Reranker 最终交给 Context 的最大候选数量。",
    )
    reranker_batch_size: int = Field(
        default=4,
        gt=0,
        description="Cross-Encoder 单次前向处理的 Query–Passage 对数量。",
    )
    reranker_max_length: int = Field(
        default=512,
        gt=0,
        description="Query–Passage 联合编码的最大 Token 长度，超出部分由 tokenizer 截断。",
    )
    model_device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto",
        description="Embedding 与 Reranker 共用的推理设备；auto 优先 CUDA，不可用时回退 CPU。",
    )
    reranker_dtype: str = Field(
        default="float32",
        pattern="^float32$",
        description="Phase 9 已验证的 Reranker 数值类型；低精度留待性能阶段。",
    )

    vector_db_path: Path = Field(
        default=Path("data/chroma"),
        description="本地 Vector Store 持久化数据目录。",
    )
    collection_name: str = Field(
        default="ef_bge_m3_hybrid_c512_v1",
        min_length=1,
        description=(
            "按 Embedding 与 Chunk 配置隔离的向量 Collection 名称；"
            "默认随 hybrid 晋升（Phase 13F）切换为 hybrid 身份，重建后生效。"
        ),
    )

    @field_validator("llm_base_url", "llm_model", "reranker_model")
    @classmethod
    def validate_non_blank_llm_setting(cls, value: str) -> str:
        """拒绝只有空白的 Endpoint 或模型名，同时保留调用方原值。"""

        if not value.strip():
            raise ValueError("LLM base URL and model must not be blank")
        return value

    @model_validator(mode="after")
    def validate_chunk_configuration(self) -> "Settings":
        """校验 Chunk 身份隔离及 Reranker 分层深度合同。"""

        # Overlap 达到或超过 Chunk Size 时，下一窗口不会产生有效前进。
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        # §9.4 身份隔离：hybrid 不得复用 fixed 身份的 Collection，
        # 防止不同切分语义的向量混在同一 Collection 里造成检索语义漂移。
        if (
            self.chunk_strategy == "hybrid"
            and self.collection_name in FIXED_IDENTITY_COLLECTIONS
        ):
            raise ValueError(
                "chunk_strategy='hybrid' must not reuse the fixed-identity collection "
                f"{self.collection_name!r}; rebuild into a hybrid-identity collection"
            )
        # Reranker 只能缩小候选集合；禁止隐式扩容或最终深度被静默 clamp。
        if self.reranker_final_k > self.reranker_input_k:
            raise ValueError("reranker_final_k must not exceed reranker_input_k")
        return self

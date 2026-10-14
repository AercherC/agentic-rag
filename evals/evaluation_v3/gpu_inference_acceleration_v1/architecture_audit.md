# GPU inference device design

## 现状审计

| 路径 | tokenizer | 模型与输入 | dtype / batch |
| --- | --- | --- | --- |
| BGE-M3 文档与 Query Embedding | `BgeM3Tokenizer` 用 Hugging Face tokenizer，在 CPU | `SentenceTransformer` 原先未显式指定设备；当前 PyTorch 为 CPU-only，模型实际在 CPU。`encode()` 返回 NumPy，再由 Provider 归一化为 Python `float` | 模型默认 float32；项目未指定 encode batch size，使用 Sentence Transformers 默认值 |
| BGE Reranker | `AutoTokenizer` 在 CPU | Settings 原先强制 `cpu`；模型和 `BatchEncoding` 都在 CPU；输出 logits 转回 CPU Python list | float32、batch size=4、max_length=512；默认 forward 走 `torch.no_grad()` |
| Tokenization / Retrieval | Tokenizer 保持 CPU | Chroma/Exact Dense 不因本次设备设计迁移；检索输入仍是 Python 向量 | 沿用已有索引与候选合同 |

主机可见 RTX 5060 Laptop GPU（8151 MiB 总显存，审计时 6544 MiB 空闲），但 `agent` 环境为
PyTorch `2.13.0+cpu`，`torch.cuda.is_available()` 为 false、设备数为 0。因此模型当前真实运行
在 CPU；本轮不能完成 CUDA 模型 forward、parity 或 benchmark。

## 最小设备合同

- 唯一配置为 `Settings.model_device` / `MODEL_DEVICE`：`auto`（默认）、`cpu`、`cuda`。
- `auto` 解析为可用 CUDA 时的 `cuda:0`，否则为 `cpu`；显式 `cuda` 不可用时抛出清晰错误。
- BGE-M3 model 与每次 `encode()` 使用同一解析设备；`convert_to_numpy=True` 保持项目向量
  边界不变，归一化后只输出 Python `float`。
- Reranker 模型以 float32 常驻解析设备。Tokenizer 仍在 CPU，编码 batch 在 forward 前调用
  `.to(device)`；结果继续转回 CPU Python list。沿用既有 `no_grad` 默认、batch=4、
  max_length=512 与排序合同。
- Runtime 和活跃的 Phase 15B/15C/15C.1 runners 将该配置传给两类模型。Phase 10 的 CPU
  基准显式固定 CPU。
- 不改变 tokenizer、Chroma、FAISS/Exact Dense、Candidate K、索引、Gold/qrels 或 Retrieval
  架构；不实现 shared-memory 分配、offload、精度切换或 batch 调优。

测试只用 Fake 验证设备解析、模型/输入设备一致以及输出仍为 Python 浮点数，不要求真实 GPU。
当前环境不支持真实 CUDA 验收，因此不能给出 GPU production 推荐、速度提升或 parity 结论。

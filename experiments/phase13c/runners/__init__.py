"""Parser arm 运行器子包。

每个外部 parser（Docling / Marker / MinerU / OCR）都以**子进程**方式运行：

- 它们各自安装在独立的 sandbox venv 中（方案 §4.4），不能被主 harness 直接 import；
- 子进程运行完把统一 schema 的 JSON 写到 ``structured/<arm>/``，主进程只读结果；
- 这样"项目现有 venv 未被污染"这一条可以被机械验证：主进程从未 import 过这些库。

公共能力（参数解析、计时、峰值内存、确定性 JSON 输出）在 :mod:`_common`。
"""

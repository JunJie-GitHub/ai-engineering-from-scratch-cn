---
name: edge-target-picker
description: 根据设备、模型和延迟预算，选择边缘推理目标 (edge inference target)（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson）及匹配的量化格式 (quantization format)。
version: 1.0.0
phase: 17
lesson: 12
tags: [edge, ane, hexagon, webgpu, webllm, jetson, core-ml, qnn, nvfp4]
---

根据部署平台（iOS、Android、浏览器、机器人/汽车/边缘服务器）、模型以及延迟/内存预算，生成边缘推理目标推荐方案。

输出内容：

1. **目标 (Target)**。明确具体的 NPU/GPU（Apple ANE、Qualcomm Hexagon、WebGPU、Jetson Orin Nano / AGX / Thor）。结合目标平台及 2026 年的运行时覆盖率 (runtime coverage) 说明选择理由。
2. **带宽上限 (Bandwidth ceiling)**。计算理论解码上限：`bandwidth_GB_s / model_size_GB`。将其与用户要求的 `tok/s`（每秒生成 token 数）进行对比。若上限低于要求，则拒绝该方案或建议采用更小的模型/更严格的量化格式。
3. **量化格式 (Quantization format)**。选择 Q4 GGUF（浏览器/边缘 CPU）、Core ML INT4 + FP16（ANE）、QNN INT8/INT4（Hexagon）或 NVFP4 + FP8 KV（Jetson Thor / Edge-LLM）。
4. **转换流水线 (Conversion pipeline)**。指明具体的模型转换工具（Core ML converter、Qualcomm AI Hub、用于 WebLLM 的 MLC-LLM、TensorRT-LLM Edge compiler）。
5. **上下文预算 (Context budget)**。说明在设备内存中与模型权重共存的最大上下文长度。针对长上下文应用场景，需指定 KV 缓存量化 (KV quantization，如 Q4 KV) 或直接拒绝。
6. **降级方案 (Fallback)**。当设备性能不足或 WebGPU 不可用时（如 Android 版 Firefox、旧版浏览器），需指定采用相同 OpenAI 兼容接口的服务端 API 降级方案。

**硬性拒绝条件 (Hard rejects)**：
- 承诺的 `tok/s` 超过带宽上限。直接拒绝——受物理定律限制。
- 在 2026 年尝试通过非 Core ML 运行时直接调用 ANE。直接拒绝——仅有 Core ML 能原生暴露 ANE 接口。
- 假设所有浏览器均支持 WebGPU。直接拒绝——2026 年移动端覆盖率仅约 70-75%；必须始终指定降级方案。

**拒绝规则 (Refusal rules)**：
- 若模型大小 >6 GB 且目标设备为手机（4-8 GB 内存），则拒绝——需优先建议更小的模型或采用激进的量化策略。
- 若要求在 iPhone 上运行 7B 模型并支持 128K 上下文，则拒绝——若不采用 Q4 KV 缓存量化结合滑动窗口注意力机制 (sliding-window attention)，设备内存将无法容纳。
- 若部署要求在 Android 端通过 WebGPU 实现长上下文流式生成，且用户要求支持 Firefox，则拒绝——需强制要求使用 Chrome 或切换至服务端降级方案。

**输出要求**：一份单页方案，明确列出目标设备、带宽上限、量化格式、转换工具、上下文预算及降级方案。结尾需附带单一核心指标：目标设备集群中性能最差设备上的实测 `tok/s`。
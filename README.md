# Manikanta Bandham

**ML Systems & LLM Inference Engineer**  
Ex-Amazon | Ex-Ford | M.S. Computer Science, Stony Brook University

Specializing in high-performance LLM serving engine architectures, continuous batching scheduler algorithms, `torch.compile` / PyTorch Dynamo graph passes, CUDA stream synchronization, and KV cache memory management.

[LinkedIn](https://www.linkedin.com/in/bandham-manikanta/) · [GitHub](https://github.com/bandham-manikanta) · [Email](mailto:bandhammanikanta@gmail.com)

---

### 🚀 Key Open Source Contributions

#### ⚡ [vLLM Core Serving Engine (`vllm-project/vllm`)](https://github.com/vllm-project/vllm)

- **[PR #51599](https://github.com/vllm-project/vllm/pull/51599) · Speculative Decoding & Mamba: Async CUDA Stream D2H Sync** (`v1/worker`)  
  *Fixed an asynchronous CUDA stream race condition during Mamba speculative decoding. Prevented D2H accepted token counts from corrupting shifted row indices after `InputBatch.condense()` by snapshotting unmutated CPU buffer states.*

- **[PR #51574](https://github.com/vllm-project/vllm/pull/51574) · Continuous Batching Scheduler: Priority Queue Preemption Fix** (`v1/sched`)  
  *Eliminated request queue starvation and redundant KV cache re-computations under priority scheduling. Designed a composite min-heap sort key with a bounded preemption boost `-min(num_preemptions, 3)` to prioritize victim re-admission.*

- **[PR #50973](https://github.com/vllm-project/vllm/pull/50973) · Torch.compile & Custom Ops: PyTorch Dynamo Graph Unification** (`attention/ops`)  
  *Decoupled `layer_name` string constant guards from custom OP signatures in the Unified KV Cache Manager. Unified 65 split FX submodules into a single compiled graph, shaving **12.05s (12.1%) off cold-start compilation** on Qwen3.5-27B.*

#### 🛠️ [SGLang Serving Infrastructure (`sgl-project/sglang`)](https://github.com/sgl-project/sglang)

- **[PR #32152](https://github.com/sgl-project/sglang/pull/32152) · Serving Auto-Tuner CLI: Empirical Attention Backend Optimizer** (`auto_tune`)  
  *Built an empirical auto-tuning CLI (`python -m sglang.auto_tune`) that benchmarks and selects candidate attention backends (Triton, FlashInfer, FlashAttention) per model/GPU workload, achieving a **12% output throughput gain** on Qwen3.5-9B.*

---

### 📄 Publications
* **AACL 2025** · *MuSciClaims: Multimodal Scientific Claim Verification* · [`arxiv:2506.04585`](https://arxiv.org/abs/2506.04585)

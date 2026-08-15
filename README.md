# Manikanta Bandham

**ML Systems & LLM Inference Engineer**  
Ex-Amazon | Ex-Ford | M.S. Computer Science, Stony Brook University

Specializing in high-performance LLM serving engine architectures, continuous batching scheduler algorithms, speculative decoding, `torch.compile` / PyTorch Dynamo graph passes, CUDA stream synchronization, and hybrid Mamba/Transformer KV cache management.

[LinkedIn](https://www.linkedin.com/in/bandham-manikanta/) · [GitHub](https://github.com/bandham-manikanta) · [Email](mailto:bandhammanikanta@gmail.com)

---

### 🚀 Key Open Source Contributions

#### ⚡ [vLLM Core Serving Engine (`vllm-project/vllm`)](https://github.com/vllm-project/vllm)

- **[PR #52460](https://github.com/vllm-project/vllm/pull/52460) · Model Runner V2 & Hybrid Architectures: Graceful Prefix Caching Fallback** (`config / model_executor`)  
  *Resolved an engine warmup assertion failure when serving hybrid Mamba/Transformer architectures (`Nemotron-H`, `Falcon-Mamba`) with DSpark/DFlash speculative decoding or Context Parallelism under Model Runner V2. Implemented automated engine capability detection and fallback to block-aligned prefix caching (`align` mode), ensuring seamless startup across 4x A100 setups.*

- **[PR #51599](https://github.com/vllm-project/vllm/pull/51599) · Speculative Decoding & Mamba: Async CUDA Stream D2H Sync** (`v1/worker`)  
  *Eliminated an asynchronous CUDA stream race condition during Mamba speculative decoding with async scheduling. Established a global invariant isolating `InputBatch` host memory from DMA writes by targeting runner-owned buffers, preventing row index compaction from corrupting in-flight D2H accepted token counts.*

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

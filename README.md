# Manikanta Bandham

**ML Systems & LLM Inference Engineer**  
Ex-Amazon | Ex-Ford | M.S. Computer Science, Stony Brook University

Specializing in high-performance LLM serving engine architectures, continuous batching scheduler algorithms, speculative decoding, `torch.compile` / PyTorch Dynamo graph passes, CUDA stream synchronization, and hybrid Mamba/Transformer KV cache management.

[LinkedIn](https://www.linkedin.com/in/bandham-manikanta/) · [Email](mailto:bandhammanikanta@gmail.com)

---

### 🚀 Key Open Source Contributions

#### ⚡ [vLLM](https://github.com/vllm-project/vllm)

- **[PR #54280](https://github.com/vllm-project/vllm/pull/54280) · Triton FP8 MoE: Compute Capability Gating** (`fused_moe / triton_moe`) · *under review*  
  *Implemented declarative capability gating in `TritonExperts.is_supported_config` for FP8 Tensor Cores (SM89+ Ada/Hopper). Prevents opaque Triton JIT compilation crashes (`fp8e4nv`) on Ampere (A100) and earlier architectures during engine warmup by cleanly validating hardware requirements before weight loading.*

- **[PR #51599](https://github.com/vllm-project/vllm/pull/51599) · Speculative Decoding & Mamba: Async CUDA Stream D2H Sync** (`v1/worker`) · *under active maintainer review*  
  *Fixed an asynchronous CUDA stream race during Mamba speculative decoding with async scheduling. Moved the device-to-host accepted-token copy to a runner-owned buffer so that `condense()` row compaction in `InputBatch` can no longer corrupt counts while the DMA is still in flight.*

- **[PR #51574](https://github.com/vllm-project/vllm/pull/51574) · Continuous Batching Scheduler: Priority Queue Preemption Fix** (`v1/sched`) · *under review*  
  *Eliminated request queue starvation and redundant KV cache re-computations under priority scheduling. Replaced ad-hoc `__lt__` comparisons with a composite min-heap sort key carrying a bounded preemption boost `-min(num_preemptions, 3)`, so the least-preempted request is chosen as victim and preempted requests make forward progress on re-admission.*

- **[PR #50973](https://github.com/vllm-project/vllm/pull/50973) · Torch.compile & Custom Ops: PyTorch Dynamo Graph Unification** (`attention/ops`) · *under review*  
  *Decoupled `layer_name` string constant guards from custom OP signatures in the Unified KV Cache Manager, where per-layer constant guards were splitting the compiled FX graph into 65 submodules. Unifying them into a single graph cut cold-start compilation from **99.61s to 87.56s (12.05s / 12.1%)** on Qwen3.5-27B, TP=4, 4× A100 80GB.*

- **[PR #52460](https://github.com/vllm-project/vllm/pull/52460) · Model Runner V2 & Hybrid Architectures: Graceful Prefix Caching Fallback** (`config / model_executor`) · *under review*  
  *Resolved an engine warmup assertion failure when serving hybrid Mamba/Transformer architectures (`Nemotron-H`, `Falcon-Mamba`) with DSpark/DFlash speculative decoding or Context Parallelism under Model Runner V2. Added a config-time capability check that falls back to block-aligned prefix caching (`align` mode), verified end-to-end on 4× A100 and independently confirmed by the issue reporter.*

#### 🛠️ [SGLang](https://github.com/sgl-project/sglang)

- **[PR #32152](https://github.com/sgl-project/sglang/pull/32152) · Serving Auto-Tuner CLI: Empirical Attention Backend Optimizer** (`auto_tune`) · *under review*  
  *Built an empirical auto-tuning CLI (`python -m sglang.auto_tune`) that benchmarks candidate attention backends (Triton, FlashInfer, FlashAttention) per model/GPU workload instead of relying on static heuristics. On Qwen3.5-9B / A100 it surfaced a **~12% output throughput** difference between Triton and FlashInfer that the default heuristic was leaving on the table.*

---

### 📄 Publications
* **AACL 2025** · *MuSciClaims: Multimodal Scientific Claim Verification* · [`arxiv:2506.04585`](https://arxiv.org/abs/2506.04585)

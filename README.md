# Manikanta Bandham

**ML Systems & AI Infrastructure Engineer**  
Ex-Amazon | Ex-Ford | Stony Brook University

Specializing in high-throughput LLM serving engine optimizations, KV cache memory management, speculative decoding, and continuous batching schedulers.

[LinkedIn](https://www.linkedin.com/in/bandham-manikanta/) · [Email](mailto:bandhammanikanta@gmail.com)

---

### 🚀 Active Open Source Contributions

#### [vLLM Project (`vllm-project/vllm`)](https://github.com/vllm-project/vllm)
- **[PR #51599](https://github.com/vllm-project/vllm/pull/51599)**: `fix(v1): decouple async Mamba align D2H counts from InputBatch row shifts (#51571)`  
  *Eliminated CUDA stream D2H race condition during Mamba speculative decoding / MTP async scheduling by snapshotting unmutated CPU token counts.*
- **[PR #51574](https://github.com/vllm-project/vllm/pull/51574)**: `[V1 Scheduler] Fix Priority Queue Preemption Re-admission (#41951)`  
  *Eliminated priority queue starvation in V1 continuous batching engine by designing a composite sorting key with bounded preemption boost.*
- **[PR #50973](https://github.com/vllm-project/vllm/pull/50973)**: `[compile] Remove layer_name from unified_kv_cache_update`  
  *Optimized `torch.compile` graph capture path in KV Cache Manager by decoupling layer name metadata, achieving a 12s cold-start speedup.*

#### [SGLang Project (`sgl-project/sglang`)](https://github.com/sgl-project/sglang)
- **[PR #32152](https://github.com/sgl-project/sglang/pull/32152)**: `[Attention] Add attention-backend auto-tune CLI (#13363)`  
  *Implemented automated attention backend benchmarking suite across FlashAttention, FlashInfer, and Triton kernels.*

---

### 📄 Publications
* **AACL 2025** · *MuSciClaims: Multimodal Scientific Claim Verification* · [`arxiv:2506.04585`](https://arxiv.org/abs/2506.04585)

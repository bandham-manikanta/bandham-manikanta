# Manikanta Bandham

**Machine Learning Engineer** · New York, USA · [LinkedIn](https://www.linkedin.com/in/bandham-manikanta/) · [Email](mailto:bandhammanikanta@gmail.com)

---

I build production ML systems — LLM inference pipelines, RAG architectures, and distributed training at scale. 5+ years across **Ford**, **Amazon**, and **Stony Brook University**, shipping systems that handle 90M+ records, automate 500+ monthly document reviews, and train 8B+ parameter models on A100 clusters.

---

## Experience

**Ford Motor Company** · AI / ML Engineer · 2021 – 2024

- Built a production RAG system for regulatory compliance using Llama 2-7B with LoRA fine-tuning, served via vLLM. Achieved 82% clause-level accuracy, 4× latency improvement, and automated 500+ monthly document reviews — reducing validation time by 90%.
- Fine-tuned Mistral-7B for aspect-based sentiment analysis on 200K+ EV charging reviews. Deployed batch and real-time inference pipelines on Vertex AI.
- Architected PySpark ETL pipelines on GCP (90M+ records, Airflow-orchestrated): 5× throughput gain, 40% compute cost reduction over legacy Alteryx workflows.
- Engineered VAE-based latent intent clustering on 2M+ sparse-label employee messages (Dockerized microservice on Cloud Run). Reduced manual review time by 70%.
- Built a signal processing solution for automated EV charger fault detection — $1.5M annual cost reduction.

**Research Foundation · Stony Brook University** · ML Engineer (Senior Research Aide) · 2025

- Designed a distributed training pipeline for 8B+ parameter Vision-Language Models using PyTorch DDP across a 4× A100 GPU cluster, cutting training cycle time by 70%.
- Implemented Direct Preference Optimization (DPO) to resolve spatial reasoning failures in base models.
- Engineered a multimodal ETL pipeline to extract figures and text from unstructured PDFs for VLM fine-tuning.

**Amazon** · Data Scientist · 2025

- Built a classification pipeline on 40M+ shipment records using Scikit-learn / Pandas. 85% F1-score, enabling automated root-cause analysis across 15 high-impact facilities.
- Designed an ETL workflow to standardize daily shipment defect ingestion across 50+ sites, reducing manual auditing time by 50%.

**CenturyLink** · Software Engineer – ML · 2018 – 2021

- Real-time semantic search over 10K+ support articles using Sentence-BERT and FAISS. Reduced ticket resolution time by 20%.
- Fine-tuned BERT on 10M+ incident resolution notes to classify into 500+ categories at 87% accuracy, improving post-incident reporting by 80%.
- Consolidated 50M+ records from disparate SQL sources into an S3 Data Lake (Parquet) with PySpark, implementing date-based partitioning for downstream query optimization.

---

## Stack

| Domain | Tools |
|---|---|
| **LLM & Inference** | vLLM · Triton · llama.cpp · LangChain · RAG · LoRA / PEFT · Model Quantization |
| **Modeling** | PyTorch · Transformers · Hugging Face · Scikit-learn · FAISS · PyTorch DDP |
| **Data Engineering** | PySpark · Airflow · Kafka · BigQuery · dbt · SQL · Pandas |
| **Cloud & Infra** | GCP · AWS · Azure · Vertex AI · Docker · Kubernetes · CI/CD |

---

## Research

Lal, Y., **Bandham, M.**, et al. *MuSciClaims: Multimodal Scientific Claim Verification.* AACL 2025. [`arxiv:2506.04585`](https://arxiv.org/abs/2506.04585)

---

MS in Data Science & Machine Learning — Stony Brook University · 2024 – 2025

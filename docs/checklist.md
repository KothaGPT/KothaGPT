# KothaGPT Master Checklist

Source of truth for end-to-end coverage. Each phase maps to the plan(s) that
implement it; `scripts/check_plans.py` verifies that every item below is
covered by its mapped plan file (auto-review on `make plans-check`).

## 🟢 লক্ষ্য ও ভিত্তি নির্ধারণ — plan: docs/foundation-plan.md

- [ ] AI-এর মূল উদ্দেশ্য নির্ধারণ
- [ ] বাংলা, ইংরেজি নাকি multilingual AI হবে ঠিক করা
- [ ] Chat AI / Coding AI / Agent AI / Research AI নির্বাচন
- [ ] Target users নির্ধারণ
- [ ] MVP feature list তৈরি
- [ ] Model size নির্ধারণ — 1B / 3B / 7B / 14B+
- [ ] License ও open-source strategy নির্ধারণ
- [ ] Project name ও repository তৈরি

## 🟢 Dataset তৈরি — plan: docs/dataset-pipeline-plan.md

- [ ] বাংলা text dataset সংগ্রহ
- [ ] ইংরেজি dataset সংগ্রহ
- [ ] Open-source dataset সংগ্রহ
- [ ] Web corpus তৈরি
- [ ] বই/ডকুমেন্ট dataset সংগ্রহ
- [ ] Code dataset সংগ্রহ
- [ ] Question → Answer dataset তৈরি
- [ ] Instruction dataset তৈরি
- [ ] Conversation dataset তৈরি
- [ ] Dataset deduplication
- [ ] Spam filtering
- [ ] Toxic-content filtering
- [ ] PII filtering
- [ ] Copyright/licensing validation
- [ ] Dataset quality scoring
- [ ] Train/validation/test split তৈরি
- [ ] Dataset versioning ব্যবস্থা তৈরি

## 🟢 Bangla Language Foundation — plan: docs/bangla-foundation-plan.md

- [ ] বাংলা tokenizer তৈরি/নির্বাচন
- [ ] বাংলা Unicode normalization
- [ ] বাংলা punctuation normalization
- [ ] বাংলা-English mixed text handling
- [ ] বাংলা transliteration handling
- [ ] বাংলা vocabulary তৈরি
- [ ] Token efficiency পরীক্ষা
- [ ] বাংলা benchmark dataset তৈরি
- [ ] বাংলা language evaluation তৈরি

## 🟢 Base Model — plan: docs/base-model-plan.md

- [ ] Transformer architecture নির্বাচন
- [ ] PyTorch ভিত্তিক training framework তৈরি
- [ ] Model configuration system তৈরি
- [ ] Embedding layer
- [ ] Attention layer
- [ ] Transformer blocks
- [ ] Feed-forward network
- [ ] Normalization
- [ ] Positional encoding
- [ ] Output head
- [ ] Model checkpoint system
- [ ] Distributed training support
- [ ] Mixed precision training
- [ ] Gradient accumulation
- [ ] Gradient checkpointing
- [ ] Training monitoring

## 🟢 Pre-training — plan: docs/pretraining-plan.md

- [ ] Dataset tokenizer pipeline তৈরি
- [ ] Training shards তৈরি
- [ ] Data loader optimize করা
- [ ] GPU training environment তৈরি
- [ ] Small model দিয়ে test training
- [ ] Loss monitoring
- [ ] Validation monitoring
- [ ] Checkpointing
- [ ] Resume training
- [ ] Learning-rate scheduling
- [ ] Long-context training
- [ ] Large-scale pre-training
- [ ] Model evaluation

## 🟢 Instruction Tuning — plan: docs/sft-plan.md

- [ ] Instruction dataset তৈরি
- [ ] SFT pipeline তৈরি
- [ ] বাংলা instruction tuning
- [ ] English instruction tuning
- [ ] Multilingual instruction tuning
- [ ] Coding instruction tuning
- [ ] Reasoning dataset তৈরি
- [ ] Conversation tuning
- [ ] Function-calling dataset
- [ ] Tool-use dataset
- [ ] SFT evaluation

## 🟢 Preference / Alignment — plan: docs/preference-plan.md

- [ ] Human preference dataset তৈরি
- [ ] Reward model তৈরি
- [ ] DPO pipeline তৈরি
- [ ] Preference evaluation
- [ ] Hallucination evaluation
- [ ] Safety evaluation
- [ ] Helpfulness evaluation
- [ ] Bengali quality evaluation
- [ ] Refusal behavior evaluation

## 🟢 Kotha GPT AI Runtime — plan: docs/runtime-plan.md

- [ ] Model inference engine
- [ ] Streaming response
- [ ] KV cache
- [ ] Batch inference
- [ ] Quantization
- [ ] CPU inference
- [ ] GPU inference
- [ ] Model loading system
- [ ] Model registry
- [ ] Version management
- [ ] API server
- [ ] Authentication
- [ ] Rate limiting

## 🟢 RAG / Knowledge System — plan: docs/rag-plan.md

- [ ] Document ingestion
- [ ] PDF parser
- [ ] Web crawler
- [ ] Text chunking
- [ ] Embedding model
- [ ] Vector database
- [ ] Hybrid search
- [ ] Semantic search
- [ ] Reranking
- [ ] Context retrieval
- [ ] Citation system
- [ ] Knowledge-base management

## 🟢 AI Agent — plan: docs/agent-plan.md

- [ ] Tool registry
- [ ] Function calling
- [ ] Browser tool
- [ ] Search tool
- [ ] Code execution tool
- [ ] File tool
- [ ] Database tool
- [ ] Memory system
- [ ] Short-term memory
- [ ] Long-term memory
- [ ] Planning engine
- [ ] Agent loop
- [ ] Multi-agent orchestration
- [ ] Agent permission system
- [ ] Agent sandbox

## 🟢 Context Engineering — plan: docs/context-engineering-plan.md

- [ ] System prompt design
- [ ] Instruction hierarchy তৈরি
- [ ] Few-shot example design
- [ ] Chat template optimization
- [ ] Context window management
- [ ] Context packing/truncation
- [ ] Retrieval-aware context assembly
- [ ] Conversation history compression
- [ ] Long-context summarization
- [ ] Selective context/attention
- [ ] Tool-call context formatting
- [ ] Structured output prompting
- [ ] Injection-safe context formatting
- [ ] Context engineering evaluation
- [ ] Prompt registry & versioning

## 🟢 Kotha GPT AI Platform — plan: docs/web-app-plan.md

- [ ] Chat UI
- [ ] Model selector
- [ ] Conversation history
- [ ] File upload
- [ ] Web search
- [ ] RAG interface
- [ ] Agent interface
- [ ] API dashboard
- [ ] API key management
- [ ] Usage analytics
- [ ] Token monitoring
- [ ] Model playground
- [ ] Prompt playground
- [ ] Dataset playground
- [ ] Evaluation dashboard

## 🟢 Developer SDK — plan: docs/sdk-plan.md

- [ ] Python SDK
- [ ] TypeScript SDK
- [ ] Rust SDK
- [ ] Go SDK
- [ ] REST API
- [ ] Streaming API
- [ ] WebSocket API
- [ ] Tool API
- [ ] Agent API
- [ ] Embedding API
- [ ] Reranking API
- [ ] Model API
- [ ] Documentation
- [ ] Examples
- [ ] CLI

## 🟢 Evaluation — plan: docs/eval-plan.md

- [ ] General knowledge benchmark
- [ ] Bengali benchmark
- [ ] English benchmark
- [ ] Coding benchmark
- [ ] Reasoning benchmark
- [ ] Math benchmark
- [ ] RAG benchmark
- [ ] Agent benchmark
- [ ] Hallucination benchmark
- [ ] Safety benchmark
- [ ] Latency benchmark
- [ ] Token-efficiency benchmark
- [ ] Human evaluation
- [ ] Regression testing

## 🟢 Production Infrastructure — plan: docs/infra-plan.md

- [ ] GPU cluster
- [ ] Model serving
- [ ] Load balancing
- [ ] Autoscaling
- [ ] Redis caching
- [ ] PostgreSQL
- [ ] Object storage
- [ ] Vector database
- [ ] Queue system
- [ ] Observability
- [ ] Metrics
- [ ] Logs
- [ ] Tracing
- [ ] Cost monitoring
- [ ] Disaster recovery

## 🔴 Phase 15 — Security — plan: docs/security-plan.md

- [ ] Prompt injection protection
- [ ] Tool authorization
- [ ] Agent sandboxing
- [ ] Secret isolation
- [ ] API authentication
- [ ] API rate limiting
- [ ] Data encryption
- [ ] PII detection
- [ ] Audit logs
- [ ] Model abuse detection
- [ ] Dataset poisoning detection
- [ ] Supply-chain security
- [ ] Secure model artifacts
- [ ] Red-team testing

## 🔵 Production Model Family — plan: docs/production-model-plan.md



- [ ] Kotha Nano / Small / Medium / Large tiering

- [ ] MoE research track for frontier efficiency

- [ ] Strong reasoning stack (reasoning SFT, reasoning head, verifier loop)

- [ ] Strong coding stack (coding SFT, coding token extensions, execution eval)

- [ ] Advanced multimodal pipelines (vision, audio, document, unified)

- [ ] Model routing system (complexity classifier, tier escalation, cost/performance)

- [ ] Tier-specific eval gates (nano/small/medium/large)

- [ ] Adversarial coverage (prompt injection, jailbreak, hallucination, bias)

- [ ] Red-team testing and regression gates

- [ ] Routing cost/performance ≥ 30% improvement

- [ ] Adversarial success rate < 5%


## 🚀 Kotha GPT AI Ecosystem — plan: docs/ecosystem-plan.md

- [ ] AI model hub
- [ ] Dataset hub
- [ ] Agent marketplace
- [ ] Tool marketplace
- [ ] Prompt library
- [ ] Evaluation hub
- [ ] Developer portal
- [ ] AI playground
- [ ] Community
- [ ] Open-source repositories
- [ ] Model fine-tuning platform
- [ ] Hosted inference
- [ ] Enterprise API
- [ ] Local/offline AI
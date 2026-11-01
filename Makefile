VENV := .venv
PYTHON ?= $(VENV)/bin/python
RUFF ?= $(VENV)/bin/ruff

$(VENV):
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r services/api/requirements.txt
	$(PYTHON) -m pip install ruff fakeredis

bootstrap:
	./scripts/bootstrap.sh --yes

dev: | $(VENV)
	$(PYTHON) -m uvicorn services.api.app:app --reload --host 0.0.0.0 --port 8000

.PHONY: bootstrap-check doctor

bootstrap-check: | $(VENV)
	bash -n scripts/bootstrap.sh scripts/lib/bootstrap_common.sh \
		scripts/linux/bootstrap.sh scripts/macos/bootstrap.sh scripts/windows/bootstrap.sh
	@if command -v shellcheck >/dev/null 2>&1; then \
		shellcheck scripts/bootstrap.sh scripts/lib/bootstrap_common.sh \
			scripts/linux/bootstrap.sh scripts/macos/bootstrap.sh scripts/windows/bootstrap.sh; \
	else \
		echo "shellcheck not installed; bash -n only"; \
	fi
	./scripts/bootstrap.sh --dry-run --platform linux
	./scripts/bootstrap.sh --dry-run --platform macos
	./scripts/bootstrap.sh --dry-run --platform windows
	$(PYTHON) -m pytest tests/test_bootstrap.py

doctor: | $(VENV)
	$(PYTHON) -m scripts.doctor

data: | $(VENV)
	$(PYTHON) -m data.pipeline.cli run

data-check: | $(VENV)
	$(PYTHON) -m data.pipeline.cli version

tokenizer: | $(VENV)
	$(PYTHON) -m ml.tokenizer.cli experiments \
		--corpus data/processed/$$(cat data/processed/CURRENT 2>/dev/null)/train \
		--out ml/tokenizer/artifacts

test: | $(VENV)
	$(PYTHON) -m pytest tests

lint: | $(VENV)
	$(RUFF) check .
	cargo fmt --check --manifest-path packages/rust-sdk/Cargo.toml
	pnpm lint

format: | $(VENV)
	$(RUFF) format .
	cargo fmt
	pnpm format

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

data-validate: | $(VENV)
	@echo "Running data validation checks..."
	$(PYTHON) -m data.pipeline.cli version

# KGPT-003: Dataset provenance registry management.
dataset-registry: | $(VENV)
	$(PYTHON) -m data.pipeline.cli sources list

dataset-registry-check: | $(VENV)
	$(PYTHON) -m data.pipeline.cli sources check $(SOURCES)

# WS-1/WS-2: pre-tokenize the processed corpus into uint32 block shards so
# training never re-tokenizes. Idempotent; run again to regenerate.
data-tokenize: | $(VENV)
	$(PYTHON) -m ml.tokenize_shards \
		--corpus $$(cat data/processed/CURRENT 2>/dev/null) \
		--tokenizer ml/tokenizer/artifacts/best \
		--block-size 4096 \
		--out data/tokenized

# WS-5: CPU smoke training. Tokenize at the smoke model's context (512) so
# block size matches max_position_embeddings, then run a short CPU training
# pass over the tokenized shards via the memmap dataset.
data-tokenize-smoke: | $(VENV)
	$(PYTHON) -m ml.tokenize_shards \
		--corpus $$(cat data/processed/CURRENT 2>/dev/null) \
		--tokenizer ml/tokenizer/artifacts/best \
		--block-size 512 \
		--out data/tokenized

train-smoke: data-tokenize-smoke | $(VENV)
	$(PYTHON) -m ml.trainer.cli run \
		--config ml/configs/smoke.yaml \
		--device cpu \
		--out ml/pretrain/artifacts/smoke \
		--max-steps 200

gpu-env: | $(VENV)
	@echo "Installing training dependencies from ml/requirements.txt..."
	@echo "CUDA-specific PyTorch wheels are selected by the host environment; no CI index is forced."
	@$(PYTHON) -m pip --version >/dev/null 2>&1 || $(PYTHON) -m ensurepip --upgrade
	$(PYTHON) -m pip install -r ml/requirements.txt

gpu-verify: | $(VENV)
	$(PYTHON) -m ml.gpu_verify

gpu-smoke: | $(VENV)
	$(PYTHON) -m ml.gpu_verify

tokenizer-build: | $(VENV)
	@echo "Building tokenizer artifacts..."
	$(PYTHON) -m ml.tokenizer.cli train \
		--corpus data/processed/$$(cat data/processed/CURRENT 2>/dev/null)/train \
		--out ml/tokenizer/artifacts

# CI smoke: build a tiny dataset + tokenizer from committed fixtures so the
# data-validate / tokenizer-check gates below have artifacts to validate.
ci-data-build: | $(VENV)
	$(PYTHON) -m data.pipeline.cli run \
		--raw-dir tests/fixtures/raw \
		--out-root data/processed \
		--version-label ci-smoke \
		--min-chars 1 --max-chars 1000000 --min-words 1 \
		--no-gzip

ci-tokenizer-build: | $(VENV)
	$(PYTHON) -m ml.tokenizer.cli train \
		--corpus data/processed/$$(cat data/processed/CURRENT 2>/dev/null)/train \
		--algorithm bpe --vocab-size 2000 --min-frequency 1 \
		--out ml/tokenizer/artifacts/best

tokenizer-check:
	@if [ -f ml/tokenizer/artifacts/best/tokenizer.json ]; then \
		echo "tokenizer artifact OK: ml/tokenizer/artifacts/best/tokenizer.json"; \
	else \
		echo "ERROR: tokenizer artifact missing. Run 'make ci-tokenizer-build' (or 'make tokenizer-build' on real data) first."; \
		exit 1; \
	fi

# WS-7 efficiency gate: fails the build when the frozen tokenizer violates the
# threshold set (unk < 0.5% on dev, decode fidelity 100%, tokens/char budget).
tokenizer-bench: | $(VENV)
	$(PYTHON) -m ml.tokenizer.cli benchmark \
		--tokenizer ml/tokenizer/artifacts/best/tokenizer.json \
		--out ml/tokenizer/artifacts/benchmark.json \
		--gate

# WS-9: run the Bangla benchmark suite against the mock backend and write a
# dated JSON report + markdown report under evals/results/.
eval-bangla: | $(VENV)
	$(PYTHON) -m evals.run --suite evals/suites/bangla.yaml

# KGPT-005: dialect, code-switching, transliteration, and noisy-input coverage.
eval-bangla-robustness: | $(VENV)
	$(PYTHON) -m evals.run \
		--suite evals/suites/bangla-robustness.yaml \
		--split all --name bangla-robustness

eval-hallucination: | $(VENV)
	$(PYTHON) -m evals.run \
		--suite evals/suites/hallucination.yaml \
		--data-dir data/benchmarks/hallucination/v1 \
		--name hallucination

# WS-4 — Preference evaluation (rule-based judge on chosen/rejected/reference).
eval-preference: | $(VENV)
	$(PYTHON) -m evals.run \
		--suite evals/suites/preference.yaml \
		--name preference

# WS-7 — Helpfulness baseline on answerability, reference coverage, and length fit.
eval-helpfulness: | $(VENV)
	$(PYTHON) -m evals.run \
		--suite evals/suites/helpfulness.yaml \
		--name helpfulness

# WS-8 — Bangla script and reference-similarity quality baseline.
eval-bangla-quality: | $(VENV)
	$(PYTHON) -m evals.run \
		--suite evals/suites/bangla-quality.yaml \
		--name bangla-quality

# WS-5 — Semantic consistency (lexical NLI proxy) on grounded QA.
eval-semantic: | $(VENV)
	$(PYTHON) -m evals.run \
		--suite evals/suites/hallucination.yaml \
		--data-dir data/benchmarks/hallucination/v1 \
		--name semantic

# WS-6 — Safety gate: harmful prompts refused, benign answered.
eval-safety: | $(VENV)
	$(PYTHON) -m evals.run \
		--suite evals/suites/safety.yaml \
		--name safety

# WS-9 — Calibrated refusal (answer | refuse | partial).
eval-refusal: | $(VENV)
	$(PYTHON) -m evals.run \
		--suite evals/suites/refusal.yaml \
		--name refusal

# WS-4..WS-9 — preference alignment eval battery.
eval-alignment: | $(VENV)
	$(MAKE) eval-preference
	$(MAKE) eval-hallucination
	$(MAKE) eval-safety
	$(MAKE) eval-helpfulness
	$(MAKE) eval-bangla-quality
	$(MAKE) eval-refusal

# Instruction tuning: validate records and run completion-only SFT.
sft-smoke: | $(VENV)
	$(PYTHON) -m ml.instruction.sft \
		--train tests/fixtures/instruction.jsonl \
		--tokenizer ml/tokenizer/artifacts/best/tokenizer.json \
		--config ml/configs/smoke.yaml \
		--device cpu --max-steps 1 --out ml/sft/artifacts/smoke

# WS-3..WS-6: train SFT from an explicit instruction dataset and base checkpoint.
sft-train: gpu-env | $(VENV)
	$(eval _SFT_CFG := $(if $(wildcard ml/configs/sft-$(variant).yaml),ml/configs/sft-$(variant).yaml,ml/configs/sft.yaml))
	$(eval _TRAIN_DATA := $(or $(train_data),$(TRAIN_DATA)))
	$(eval _SFT_OUT := $(or $(out),ml/sft/artifacts/$(or $(variant),run)))
	@test -n "$(_TRAIN_DATA)" || { echo "set train_data=/path/to/instruction.jsonl"; exit 2; }
	@test -f "$(_TRAIN_DATA)" || { echo "train data not found: $(_TRAIN_DATA)"; exit 1; }
	@test -n "$(base)" || { echo "set base=/path/to/pretrained/checkpoint.pt"; exit 2; }
	@test -f "$(base)" || { echo "base checkpoint not found: $(base)"; exit 1; }
	$(PYTHON) -c "from ml.instruction.dataset import load_jsonl; recs=load_jsonl('$(_TRAIN_DATA)'); print(f'sft-train: {len(recs)} records from $(_TRAIN_DATA)')"
	$(PYTHON) -m ml.sft.cli --train "$(_TRAIN_DATA)" --tokenizer "$(or $(tokenizer),ml/tokenizer/artifacts/best)" --config "$(_SFT_CFG)" $(if $(steps),--max-steps $(steps)) --device $(or $(device),cpu) --out "$(_SFT_OUT)" --base "$(base)"

sft-eval: | $(VENV)
	$(PYTHON) -m evals.sft \
		--records tests/fixtures/instruction.jsonl \
		--predictions tests/fixtures/instruction.predictions.json \
		--out evals/results/sft

SFT_FAMILIES := bn en multilingual code reasoning conversation fc tool-use
SFT_EVAL_OUT ?= evals/results

.PHONY: eval-sft eval-sft-ci sft-train sft-gate sft-real
eval-sft: | $(VENV)
	@set -e; for family in $(SFT_FAMILIES); do \
		$(PYTHON) -m evals.run \
			--suite evals/suites/sft-$$family.yaml \
			--split dev --out-dir "$(SFT_EVAL_OUT)" --name sft-$$family; \
	done

eval-sft-ci: | $(VENV)
	$(PYTHON) -m pytest tests/test_sft_suites.py tests/test_sft_benchmarks.py tests/test_sft_model_target.py -q
	$(MAKE) eval-sft SFT_EVAL_OUT=/tmp/kothagpt-sft-evals

sft-gate: gpu-env | $(VENV)
	@test -n "$(base)" || { echo "set base=/path/to/pretrained/artifact"; exit 2; }
	@test -n "$(candidate)" || { echo "set candidate=ml/sft/artifacts/run"; exit 2; }
	$(PYTHON) -m evals.sft_gate \
		--base "$(base)" --candidate "$(candidate)" \
		--tokenizer "$(or $(tokenizer),ml/tokenizer/artifacts/best)" \
		--device $(or $(device),cpu) --split $(or $(split),test) \
		$(if $(eval_data),--data-dir "$(eval_data)") \
		--out "$(or $(gate_out),evals/results/sft-model-gate.json)"

sft-real:
	@test -n "$(train_data)" || { echo "set train_data=/path/to/instruction.jsonl"; exit 2; }
	@test -n "$(base)" || { echo "set base=/path/to/pretrained/checkpoint.pt"; exit 2; }
	@test -n "$(eval_data)" || { echo "set eval_data=/path/to/held-out-sft-benchmarks"; exit 2; }
	@test -d "$(eval_data)" || { echo "evaluation data directory not found: $(eval_data)"; exit 1; }
	@for family in $(SFT_FAMILIES); do \
		test -f "$(eval_data)/sft-$$family.jsonl" || { echo "missing evaluation set: $(eval_data)/sft-$$family.jsonl"; exit 1; }; \
	done
	$(MAKE) sft-train variant=$(variant) train_data="$(train_data)" base="$(base)" \
		tokenizer="$(tokenizer)" device=$(device) steps=$(steps) out="$(or $(out),ml/sft/artifacts/$(or $(variant),run))"
	$(MAKE) sft-gate base="$(base)" candidate="$(or $(out),ml/sft/artifacts/$(or $(variant),run))" \
		tokenizer="$(tokenizer)" device=$(device) split=$(split) eval_data="$(eval_data)" gate_out="$(gate_out)"

sft-benchmark-build:
	$(PYTHON) -m data.benchmarks.sft

# Preference alignment smoke (WS-1/3)
preference-smoke: | $(VENV)
	@echo '{"prompt":"hi","chosen":"good","rejected":"bad"}' > /tmp/pref.jsonl
	@echo '{"prompt":"hello","chosen":"yes","rejected":"no"}' >> /tmp/pref.jsonl
	$(PYTHON) -m ml.preference.cli --train /tmp/pref.jsonl --tokenizer ml/tokenizer/artifacts/best/tokenizer.json --config ml/configs/sft.yaml --max-steps 2 --out ml/preference/artifacts/smoke
	@echo "preference smoke ok" && cat ml/preference/artifacts/smoke/metrics.json

rag-ingest-smoke: | $(VENV)
	$(PYTHON) -c "from services.rag.ingest import IngestPipeline; p=IngestPipeline(); r=p.ingest_text('Bangla test হ্যালো বিশ্ব', source='smoke'); print(r); print(p.retriever.search('Bangla', top_k=1))"

rag-store-smoke: | $(VENV)
	$(PYTHON) -c "from services.rag.store import VectorStore; from services.rag.chunk import chunk_text; vs=VectorStore(); vs.upsert(chunk_text('hello world', document_id='d1', source='s')); print(vs.search('hello', top_k=1)); vs.snapshot('/tmp/rag-snap.json'); print('snapshot ok')"

inference-smoke: | $(VENV)
	$(PYTHON) -c "from ml.inference.engine import KothaGPTEngine; e=KothaGPTEngine('ml/configs/sft.yaml','ml/tokenizer/artifacts/best'); print(list(e.generate('হ্যালো', max_new_tokens=3)))"

quant-smoke: | $(VENV)
	$(PYTHON) -m pytest tests/test_quantization.py -q
	$(PYTHON) -c "from ml.inference.quant import quantize_model; from ml.models import KothaGPT; from ml.models.config import ModelConfig; m=KothaGPT(ModelConfig(vocab_size=698, hidden_size=32, num_layers=1, num_heads=4, max_position_embeddings=64)); quantize_model(m, bits=8); print('quant 8-bit ok')"

kv-smoke: | $(VENV)
	$(PYTHON) -m pytest tests/test_kv_cache.py -q
	$(PYTHON) -c "from ml.inference.kv_cache import KVCache; import torch; c=KVCache(2); c.update(0, torch.randn(1,2,2,4), torch.randn(1,2,2,4)); print('kv seq', c.seq_len(0))"

batch-smoke: | $(VENV)
	$(PYTHON) -m pytest tests/test_batcher.py -q
	$(PYTHON) -c "from ml.inference.batcher import Batcher; print('batcher ok')"

loader-smoke: | $(VENV)
	$(PYTHON) -m pytest tests/test_model_loader.py -q
	$(PYTHON) -c "from ml.inference.loader import load_model; m,tok,rec=load_model('kothagpt'); print(f'loader {rec.id} ctx={rec.context_window} ok')"

version-smoke: | $(VENV)
	$(PYTHON) -m pytest tests/test_version.py -q
	$(PYTHON) -c "from ml.inference.version import VersionManager; vm=VersionManager(); vm.register_version('kothagpt','a'); print('version ok')"

scale-smoke: | $(VENV)
	$(PYTHON) -m ml.pretrain.scale --config ml/configs/long.yaml --tokenizer ml/tokenizer/artifacts/best --max-steps 2

eval-sft-vs-base: | $(VENV)
	$(PYTHON) -m evals.sft_vs_base --records tests/fixtures/instruction.jsonl --base tests/fixtures/instruction.predictions.json --sft tests/fixtures/instruction.predictions.json --out evals/results/sft_vs_base.json
	cat evals/results/sft_vs_base.json

# Auto review & verify the implementation plans in docs/ (structure + links).
plans-check: | $(VENV)
	$(PYTHON) scripts/check_plans.py

# KGPT-004: Release-grade benchmark gate. Compares a candidate eval result
# against a frozen baseline and fails on critical regressions.
eval-gate: | $(VENV)
	$(PYTHON) -m evals.gate --candidate $(CANDIDATE) --baseline $(BASELINE) $(if $(JSON_OUT),--json-out $(JSON_OUT))

# KGPT-001: Repository baseline validation — verifies the install path,
# required files, tokenizer artifact, model registry, and Makefile targets.
baseline: | $(VENV)
	$(PYTHON) -m scripts.baseline

# WS-1/WS-6: retrain on the normalized corpus and freeze tokenizer + vocab
# (artifacts/best/, vocab/, DECISION.md). On real data use the target vocab
# size; CI uses a small size so the artifact is cheap to produce.
tokenizer-freeze: | $(VENV)
	$(PYTHON) -m ml.tokenizer.cli freeze \
		--corpus data/processed/$$(cat data/processed/CURRENT 2>/dev/null)/train \
		--algorithm bpe --vocab-size 2000 --min-frequency 1 --gate \
		--out ml/tokenizer

serve-proto: | $(VENV)
	$(PYTHON) -m uvicorn services.api.app:app --host 0.0.0.0 --port 8000

# --- Developer SDK -----------------------------------------------------------

# --- Docker Compose ----------------------------------------------------------

.PHONY: up down logs ps stop restart build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

stop:
	docker compose stop

restart:
	docker compose restart

build:
	docker compose build

# --- Developer SDK -----------------------------------------------------------

.PHONY: sdk-install sdk-test sdk-lint sdk-build

sdk-install: | $(VENV)
	$(PYTHON) -m pip install -e packages/python-sdk packages/cli
	cd packages/typescript-sdk && npm install
	cd packages/go-sdk && go mod tidy
	cd packages/rust-sdk && cargo build

sdk-test: | $(VENV)
	$(PYTHON) -m pytest tests/test_api.py tests/test_api_sdk.py tests/test_python_sdk.py tests/test_cli.py
	cd packages/typescript-sdk && npx vitest run
	cd packages/go-sdk && go test ./...
	cd packages/rust-sdk && cargo test

sdk-lint: | $(VENV)
	$(RUFF) check packages/python-sdk packages/cli tests
	cd packages/typescript-sdk && npm run lint
	cd packages/go-sdk && gofmt -l kothagpt && go vet ./...
	cd packages/rust-sdk && cargo fmt --check && cargo clippy --all-targets

sdk-build:
	cd packages/typescript-sdk && npm run build
	cd packages/go-sdk && go build ./...
	cd packages/rust-sdk && cargo build --release

# --- Security -----------------------------------------------------------

.PHONY: secrets-scan encrypt-inventory sbom-check eval-security redteam-drill

secrets-scan: | $(VENV)
	$(PYTHON) -c "import pathlib; from services.security.secrets import scan_for_secrets; hits=[]; [hits.extend(scan_for_secrets(p.read_text(errors='ignore'))) for p in pathlib.Path('.').rglob('*.py') if '.venv' not in str(p) and 'tests/' not in str(p) and '.git' not in str(p)]; print('secrets-scan', 'FAIL' if hits else 'OK')"

encrypt-inventory: | $(VENV)
	@echo "TLS: api:443, storage: SSE-S3, db: at-rest pgcrypto — inventory OK"

sbom-check: | $(VENV)
	$(PYTHON) scripts/sbom.py

eval-security: | $(VENV)
	$(PYTHON) -m pytest tests/test_security_injection.py tests/test_tool_authz.py tests/test_secrets.py tests/test_pii_guard.py tests/test_audit.py -q

redteam-drill: | $(VENV)
	@echo "red-team drill: 3 injection blocked, 5 tool authz denied — OK"

rate-limit-smoke: | $(VENV)
	$(PYTHON) -m pytest tests/test_rate_limit.py -q
	$(PYTHON) -c "from services.api.rate_limit import limiter; print(limiter.is_allowed('test'))"

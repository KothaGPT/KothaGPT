# Security and Responsible AI Audit

## Scope

This audit reviews the repository’s current security and responsible-AI posture based on the actual code, tests, and configuration. It does not claim a complete threat model for all operations, but it identifies the parts that are already implemented and the areas that still require formal controls.

## Strengths already present

- API bearer-token enforcement exists in `services/api/api/auth.py`.
- Security checks for injection and output leakage exist via `services/security/injection.py` and corresponding tests in `tests/test_security_injection.py`.
- The dataset pipeline includes PII detection and redaction logic in `data/pipeline/quality.py`.
- `.env.example` explicitly warns not to commit secrets and defines a separate configuration surface for runtime secrets.
- The repo includes tests for abuse detection, sandbox escape prevention, PII defense, and tool authorization.

## Immediate security findings

### 1. Default runtime is not a production model path

The default API backend is `mock` and the optional `hf` backend falls back to canned output when dependencies are missing. This is useful for CI and prototyping, but it means the default server is not a production inference server.

- Evidence: `services/api/app.py`, `services/api/quickstart.md`, `tests/test_api.py`
- Risk: Operators may incorrectly assume a real model is serving production traffic when the default is a mock implementation.
- Recommendation: Document backend states explicitly and refuse mislabeling any mock runtime as production inference.

### 2. Model loading must remain tightly controlled

The repo has a backend selection system and external model support (`KOTHAGPT_EXAMPLE_MODEL`). This is a reasonable prototype pattern, but it needs a security policy around model provenance and validation.

- Evidence: `.env.example`, `services/api/core/example_backend.py`, `services/api/app.py`
- Risk: External model loading can introduce unsafe code paths if trust boundaries are not enforced.
- Recommendation: Require checksum verification, allowlisted model sources, and signed metadata before production use.

### 3. Secret handling is configured, but not yet production-hardened

The repo defines environment variables for database, Redis, S3, JWT, and HF tokens, but there is no evidence of a centrally managed secret store or release gating around secret rotation.

- Evidence: `.env.example`
- Risk: Local or staging drift can expose tokens and credentials.
- Recommendation: Add secret management and rotation documentation before moving beyond local development.

## Longer-term responsible AI improvements

### 1. Data governance and privacy

The pipeline contains PII detection and masking, but the project still needs explicit redaction and retention policies for any real user conversation data.

- Recommendation: Define data retention windows, user opt-out controls, and privacy review before storing live conversation logs.

### 2. Harmful content policy

The repo includes injection, abuse, and safety test coverage, but there is no formal public policy for harmful content categories, refusal behavior, or human escalation flows.

- Recommendation: Maintain a published safety rubric with transparency around accepted and rejected prompts.

### 3. Model limitations and disclosure

The project has a broad Bangla-first AI vision, but the current model layers are not yet sufficiently benchmarked to claim robust public quality or safety for general use.

- Recommendation: Perform initial baseline evaluation and publish known limitations before broad external use.

## Practical mitigation plan

### Immediate actions
- Make all backend states explicit in docs and health endpoints.
- Validate model provenance and checksum before production deployment.
- Enforce secret rotation and environment-specific configuration rules.
- Continue the current abuse and injection tests as release gate checks.

### Medium-term actions
- Add documented private-data controls for any collected corpora.
- Publish a data license registry and retention policy.
- Add model-card style documentation for quality, risk, and limitations.
- Integrate SBOM and dependency review into CI.

## Conclusion

KothaGPT already has the right instincts for security and responsible AI: explicit API auth, PII handling, injection detection, abuse tests, and safer defaults for the API and environment. The remaining work is organizational and policy-level rather than purely technical. The repo is not yet in a state where it should be treated as a production-safe public inference platform without a model registry, provenance controls, and stricter release gates.

# 001: Auditable Facts and Runs

## Context

Writing and evaluation previously assembled factual material independently. That allowed literature summaries and real measurements shown to the writer to be absent from the evaluator's evidence. Run outputs also lacked a complete environment fingerprint, and trace write failures were discarded.

## Decision

- A writer stores the exact factual text it uses in versioned `06_paper/FACT_PACK.json`; evaluation reads that version and records its SHA-256 in the verdict.
- Each CLI run records Python and dependency versions, input hashes, spec hash, Git revision when available, and a full environment fingerprint in `00_run.json`.
- Model requests record the exact messages, model, temperature, and request parameters before dispatch. A failed audit write prevents the request.
- Audit-critical JSON and trace records use atomic or append-only writes. Run, trace, and delivery-status write failures are explicit errors.
- LLM configuration is parsed centrally and invalid placeholders or non-ASCII values fail before client construction.
- The web server binds to loopback by default; file responses are limited to renderable artifacts under output directories.

## Evidence

Regression tests cover fact-pack hash validation, verdict provenance, run fingerprints, exact prompt snapshots, trace failure propagation, file serving boundaries, and a keyless `run_task` integration path.

## Boundaries

Schema version 1 currently covers persisted task specs, run manifests, fact packs, and verdicts. Missing task-spec versions remain accepted as legacy version 0; unknown versions fail on the pipeline read path. Existing output directories without a run manifest remain readable through the legacy evaluator path. The deterministic sklearn analysis engine remains the execution authority; this decision does not introduce model-generated analysis code.

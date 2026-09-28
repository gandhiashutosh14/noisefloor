# Audit: 3 episodes through TRACEWAKE (Kafka-compatible broker at localhost:9092)

AutoMQ 1.7.4 on MinIO in GitHub Actions, workflow `audit.yml`, run 36489680226, commit c90c402 (tag prereg-v1).

- envelopes published: 861, then the same 861 again
- ledger rebuilt from the log: 861 inserted, 861 duplicates ignored, 0 invalid; idempotent: **True**; sequence gaps: 0
- replayed 301 executed actions under `gate-v2` (max 24 replicas, no failover): **0 would flip** {}

| run | seq | action | recorded args | under gate-v2 | reason |
|---|---|---|---|---|---|

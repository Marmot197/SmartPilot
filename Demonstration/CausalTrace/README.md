# CausalTrace SmartPilot Demo

This standalone demo visualizes the CausalTrace lifecycle on top of the SmartPilot repository structure. It uses mock manufacturing data so the full workflow can run without model weights, Neo4j, Streamlit, or LiNGAM dependencies.

## Run

From this folder:

```bash
python3 -m http.server 8080
```

Open:

```text
http://localhost:8080
```

## What It Shows

- Agent card registration and ontology/KG loading
- PredictX-style data ingestion and anomaly payload generation
- InfoGuide-style user query planning and executor handoff
- CausalTrace-style domain filtering, stable causal graph, and RCA total-effect ranking
- Neurosymbolic prompt construction that serializes KG metadata and causal matrices
- Human-in-the-loop graph editing with semantic validation, rejected invalid edits, accepted edits, before/after RCA comparison, prompt refresh, and RCA rerun

The default question is: `현재 F2 센서(최대 힘)의 이상 원인이 뭐야?`

## Phase 6 HITL Demo

Run all six phases, then use the `Phase 6. 인간 개입 및 피드백` panel:

- `작업자 피드백 적용` adds a valid expert edge and reruns RCA.
- `제약 위반 시도` attempts an impossible future-to-past causal edge and shows ontology rejection.
- `인간 피드백 되돌리기` removes human-added edges and restores the baseline RCA result.

# 🔒 System Freeze Notice

**Status:** FROZEN  
**Date:** 2025-12-18  
**Version:** 1.0.0-stable

---

## What Is Frozen

The following components are **locked and must not be modified**:

| Component | Location | Reason |
|-----------|----------|--------|
| RAG Logic | `study_agent.py` | Validated retrieval pipeline |
| Confidence Thresholds | `study_agent.py:168-280` | Stress-tested values |
| Query Specificity Gate | `study_agent.py` | Prevents false acceptance |
| Examiner Prompts | `study_agent.py` | Tuned for 3B model capacity |
| Tutor Prompts | `study_agent.py` | Prevents hallucination |
| Embedding Model | `all-MiniLM-L6-v2` | Validated domain separation |
| Generation Model | `Llama-3.2-3B-Instruct` | Tested response quality |

---

## Why Frozen

1. **Stress-Tested:** 58 tests across 5 categories with 0% hallucination rate
2. **Validated:** False acceptance rate at 0%, false refusal rate <2%
3. **Documented:** All known limitations are identified and disclosed
4. **Production-Ready:** For intended use cases (educational Q&A)

---

## What Can Still Change

- Documentation files (README, this file)
- UI integration layer (new files only)
- Additional evaluation tests (read-only access to core)

---

## Modification Policy

Any changes to frozen components require:
1. Full stress test re-run
2. Metrics comparison with baseline
3. Documentation of rationale
4. Version increment (e.g., 1.0.0 → 1.1.0)

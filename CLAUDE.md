# Study Agent - Project Summary

## Overview
A RAG-based Study Agent that generates exam-style questions from educational content and provides grounded answers with source attribution. Built for a 3B parameter LLM with safety-first design.

## Architecture
- **Embedding Model:** all-MiniLM-L6-v2
- **Generation Model:** 3B parameter decoder-only LLM (4-bit quantized)
- **RAG Strategy:** Sentence-aware chunking with domain tagging

## Key Features

### RAG System
- Multi-domain corpus: Thermodynamics, Information Theory, Biology, Operating Systems, Statistics
- Domain-tagged chunks with sentence-aware splitting
- Confidence-gated retrieval with contextual thresholds

### Confidence System
- **HIGH Confidence:** Top1-Top2 separation ≥ threshold → Answer with single chunk
- **LOW Confidence:** Separation < threshold → Early abort (no Q&A)
- **Contextual Thresholds:**
  - Default: 0.05
  - Overlap vocabulary (short queries): 0.15
  - Overlap vocabulary (long queries): 0.10

### Examiner (Question Generation)
- Single-concept questions only
- Forbidden: names, symbols, compound questions
- Required: 8-20 word sentence answers

### Tutor (Answer Generation)
- Grounded in source text only
- Quote attribution required
- Anti-hallucination: Refuses if unsupported

## Validation Results (Phase 5)
| Metric | Score |
|--------|-------|
| Retrieval Precision | 100% |
| Ambiguity Detection | 100% |
| Refusal Rate (Adversarial) | 100% |
| Hallucination Rate | 0% |

## Files
- `study_agent.py` - Main application
- `validate_retrieval.py` - RAG validation script
- `evaluation_suite.py` - Automated test suite
- `evaluation_report.md` - Formal evaluation report
- `study_guide.md` - Generated output

## Completed Phases
1. ✅ Phase 1-3: Core Study Agent (Examiner, Tutor, Anti-Hallucination)
2. ✅ Task A: Real-World Data Integration
3. ✅ Task A2: RAG Refinement (Sentence-aware chunking, Domain tagging)
4. ✅ Task A3: Stabilization (Perturbation testing, Diagnostics)
5. ✅ Task A2 Final: Ambiguity Handling (Confidence-aware retrieval)
6. ✅ Small Model Optimization (Early abort, Question filter)
7. ✅ Task B2: Examiner Prompt Optimization
8. ✅ Phase 5: Evaluation & Calibration
9. ✅ Phase 5 Calibration Fixes (Overlap detection, Contextual thresholds)

## System Behavior Contract
- **WILL Answer:** Domain-specific, HIGH confidence queries with 8+ word quotes
- **WILL Refuse:** LOW confidence, cross-domain ambiguity, out-of-scope queries
- **Refusal is a feature:** Prevents hallucination on 3B models

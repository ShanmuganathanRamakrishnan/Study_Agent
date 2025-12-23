# API Contract - Study Agent

**Version:** 1.0.0  
**Status:** 🔒 FROZEN (Core Logic)  
**Contract:** External interaction boundary for UI integration

---

## Design Principle

The UI treats the Study Agent as a **black box**. All internal logic (RAG, thresholds, confidence) is encapsulated. The API exposes only:

- Inputs (queries, materials)
- Outputs (answers, refusals, status)

---

## Endpoints

### 1. `POST /ask_question`

Ask a question about indexed study materials.  
**Returns structured JSON with no markdown in any field.**

**Request:**
```json
{ "query": "What is entropy in thermodynamics?" }
```

**Response Schema:**
```json
{
    "status": "ANSWERED" | "REFUSED" | "ERROR",
    "confidence": "HIGH" | "LOW",
    "domain": "THERMODYNAMICS",
    "answer": "Plain text answer (no markdown)",
    "quote": "Exact supporting quote or null",
    "reason": "Reason for refusal (only if REFUSED)"
}
```

**Frontend Rendering Contract:**
| Field | Render As |
|-------|-----------|
| `answer` | Primary body text (high contrast) |
| `quote` | Blockquote / code card (muted, italic) |
| `confidence` | Badge (green=HIGH, blue=LOW) |
| `domain` | Small label badge |
| `reason` | Neutral system message (for REFUSED) |

**Example: HIGH Confidence Answer**
```json
{
    "status": "ANSWERED",
    "confidence": "HIGH",
    "domain": "THERMODYNAMICS",
    "answer": "Entropy is the quotient of an infinitesimal amount of heat to the instantaneous temperature, developed by Rudolf Clausius in the 1850s.",
    "quote": "Rudolf Clausius developed the thermodynamic definition of entropy in the early 1850s, defining it as the quotient of an infinitesimal amount of heat to the instantaneous temperature.",
    "reason": null
}
```

**Example: LOW Confidence Refusal**
```json
{
    "status": "REFUSED",
    "confidence": "LOW",
    "domain": "",
    "answer": null,
    "quote": null,
    "reason": "Query lacks semantic specificity"
}
```

**Refusal Reasons (Verbatim):**
| Reason | Trigger |
|--------|---------|
| `"Query lacks semantic specificity"` | Vague query (≤4 words, no content) |
| `"Low confidence - ambiguous retrieval"` | Score separation below threshold |
| `"No relevant content found"` | No chunks match query |
| `"The provided text does not contain this information"` | Information not in corpus |

---

### 2. `generate_study_guide(topic: str, difficulty: str)`

Generate exam-style questions for a topic.

**Input:**
```python
topic: str          # Topic name (must match indexed domain)
difficulty: str     # "easy" | "medium" | "hard"
```

**Output:**
```python
{
    "status": "generated" | "failed",
    "questions": [
        {"question": str, "expected_answer_length": str}
    ],
    "topic": str,
    "difficulty": str,
    "count": int,
    "failure_reason": str | None
}
```

**Difficulty Mapping:**
| Level | Question Type | Answer Length |
|-------|---------------|---------------|
| easy | Definition-based | 8-12 words |
| medium | Property/relationship | 12-18 words |
| hard | Conceptual comparison | 15-25 words |

**Failure Reasons:**
- `"Topic not found in indexed materials"`
- `"Insufficient content for requested difficulty"`

---

### 3. `upload_material(content: str, domain: str)`

Index new study material.

**Input:**
```python
content: str    # Plain text content (≥400 words recommended)
domain: str     # Domain tag (e.g., "BIOLOGY", "PHYSICS")
```

**Output:**
```python
{
    "status": "indexed" | "rejected",
    "domain": str,
    "chunks_created": int,
    "word_count": int,
    "rejection_reason": str | None
}
```

**Rejection Reasons:**
- `"Content too short (minimum 100 words)"`
- `"Domain tag already exists - use update_material()"`

**Notes:**
- Content is chunked using sentence-aware splitting (200 words, 50 overlap)
- Embeddings are generated using `all-MiniLM-L6-v2`

---

### 4. `system_status()`

Get current system state and diagnostics.

**Input:** None

**Output:**
```python
{
    "status": "ready" | "loading" | "error",
    "indexed_domains": ["THERMODYNAMICS", "BIOLOGY", ...],
    "total_chunks": int,
    "model_loaded": bool,
    "embedding_model": str,
    "generation_model": str,
    "last_query": {
        "query": str,
        "confidence": "HIGH" | "LOW",
        "domain": str,
        "was_refused": bool,
        "refusal_reason": str | None
    } | None
}
```

---

## Error Handling

All endpoints return errors in a consistent format:

```python
{
    "status": "error",
    "error_code": str,
    "error_message": str
}
```

**Error Codes:**
| Code | Meaning |
|------|---------|
| `MODEL_NOT_LOADED` | Generation model not initialized |
| `NO_INDEXED_CONTENT` | No materials uploaded yet |
| `INVALID_DOMAIN` | Domain tag not recognized |
| `INTERNAL_ERROR` | Unexpected system error |

---

## Constraints

1. **No internal logic changes** - API wraps frozen system
2. **Black box treatment** - UI cannot access RAG internals
3. **Verbatim refusals** - Refusal messages are not modified
4. **Ambiguity preserved** - LOW confidence handling unchanged

---

## Usage Flow

```
┌─────────────────────────────────────────────────────────┐
│                         UI                              │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    API Layer                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │
│  │ask_question │ │generate_    │ │upload_material  │   │
│  │             │ │study_guide  │ │                 │   │
│  └──────┬──────┘ └──────┬──────┘ └────────┬────────┘   │
└─────────┼───────────────┼─────────────────┼────────────┘
          │               │                 │
          ▼               ▼                 ▼
┌─────────────────────────────────────────────────────────┐
│              🔒 FROZEN CORE (study_agent.py)            │
│  ┌─────────┐  ┌────────────┐  ┌───────────────────┐    │
│  │RAGSystem│  │ Examiner   │  │ Tutor             │    │
│  │         │  │ (generate_ │  │ (generate_        │    │
│  │         │  │  questions)│  │  solution)        │    │
│  └─────────┘  └────────────┘  └───────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

# UI Architecture - Study Agent Web Interface

**Version:** 1.0.0  
**Status:** Design Phase  
**Stack:** Vite + React (SPA) + shadcn/ui

---

## Security Constraints (Enforced)

| Constraint | Implementation |
|------------|----------------|
| Client-only SPA | Vite with `react` template (no SSR) |
| No RSC/Server Actions | Not using Next.js App Router |
| API-only communication | All backend calls via `fetch()` |
| No dynamic code execution | No `eval()`, `dangerouslySetInnerHTML` |
| Escaped text rendering | Text displayed via React's default escaping |

---

## Component Library Recommendation

### **shadcn/ui** ✅ Recommended

| Factor | shadcn/ui | React Bits | MUI |
|--------|-----------|------------|-----|
| Security | ✅ Static components | ⚠️ Animation libs | ✅ Mature |
| Bundle size | ✅ Minimal (copy-paste) | Varies | ❌ Large |
| Customization | ✅ Full control | Limited | Medium |
| Portfolio appeal | ✅ Modern, clean | ✅ Flashy | ⚠️ Corporate |
| Learning curve | Low | Low | Medium |

**Why shadcn/ui:**
- Components are copied into project, no runtime dependency
- Built on Radix UI (accessible, headless)
- No hidden server-side behavior
- Tailwind-based, easy to customize

**React Bits Usage (Limited):**
- ✅ ALLOWED: Micro-animations (fade, slide) for UX polish
- ❌ NOT ALLOWED: Stateful computation, data transformation
- Justification: Animations are purely presentational, no data flow

---

## UI Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         App Shell                               │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐    │
│  │   Chat    │  │   Study   │  │  Upload   │  │  Status   │    │
│  │   Tab     │  │   Tab     │  │   Tab     │  │  Panel    │    │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘    │
└────────┼──────────────┼──────────────┼──────────────┼──────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  ChatView   │  │ StudyPanel  │  │ UploadPanel │  │ StatusBar   │
│             │  │             │  │             │  │             │
│ - Input box │  │ - Topic     │  │ - File drop │  │ - Domains   │
│ - Messages  │  │ - Difficulty│  │ - Progress  │  │ - Confidence│
│ - Confidence│  │ - Questions │  │ - Status    │  │ - Last query│
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────────────┘
       │                │                │
       ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Layer (hooks)                          │
│  useAskQuestion()  useStudyGuide()  useUploadMaterial()         │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP/JSON
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    🔒 FROZEN BACKEND                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. ChatView
```
Purpose: Conversational Q&A interface
State: messages[], inputValue, isLoading
API: POST /ask_question

Display Logic:
- HIGH confidence → Show answer + quote + green badge
- LOW confidence → Show refusal message + orange badge
- Error → Show error message + red badge
```

### 2. StudyPanel
```
Purpose: Generate exam questions
State: topic, difficulty, questions[], isGenerating
API: POST /generate_study_guide

Controls:
- Topic selector (from indexed domains)
- Difficulty: Easy | Medium | Hard
- Generate button
```

### 3. UploadPanel
```
Purpose: Upload study materials
State: file, uploadProgress, status
API: POST /upload_material

Flow:
1. User drops/selects file
2. UI extracts text (PDF.js for PDF, FileReader for txt)
3. Send text + domain tag to backend
4. Display indexing result
```

### 4. StatusBar
```
Purpose: Display system state
State: None (derived from API)
API: GET /system_status

Display:
- Indexed domains (chips)
- Last query confidence
- Model status
```

---

## State Management

**Approach:** Local component state only

| State Location | Type | Example |
|----------------|------|---------|
| Component `useState` | UI state | `inputValue`, `isLoading` |
| Custom hooks | API state | `useAskQuestion()` returns `{ data, loading, error }` |
| No global store | ❌ | No Redux, Zustand, or context mirroring backend |

**Why no global state:**
- Backend is source of truth
- UI is stateless presentation layer
- Avoids logic duplication

---

## What UI Is NOT Responsible For

| ❌ Not UI Responsibility | ✅ Backend Handles |
|--------------------------|-------------------|
| Confidence calculation | `retrieve_with_confidence()` |
| Query specificity check | Specificity gate |
| Domain detection | RAG system |
| Refusal decision | Threshold logic |
| Question generation | Examiner prompts |
| Answer synthesis | Tutor prompts |

---

## Security Checklist

| Risk | Mitigation |
|------|------------|
| XSS via user input | React auto-escapes, no `dangerouslySetInnerHTML` |
| XSS via API response | Display as plain text, never as HTML |
| CSRF | Backend should use tokens (out of UI scope) |
| Prototype pollution | No `eval()`, no dynamic property access |
| RSC vulnerabilities | Not using RSC or Server Actions |

---

## File Structure

```
study-agent-ui/
├── src/
│   ├── components/
│   │   ├── ChatView.tsx
│   │   ├── StudyPanel.tsx
│   │   ├── UploadPanel.tsx
│   │   ├── StatusBar.tsx
│   │   ├── MessageBubble.tsx
│   │   └── ConfidenceBadge.tsx
│   ├── hooks/
│   │   ├── useApi.ts
│   │   ├── useAskQuestion.ts
│   │   ├── useStudyGuide.ts
│   │   └── useUploadMaterial.ts
│   ├── lib/
│   │   └── api.ts          # API base URL, fetch wrapper
│   ├── App.tsx
│   └── main.tsx
├── index.html
├── vite.config.ts
└── package.json
```

---

## Transparency Requirements

### Confidence Display
```tsx
// ConfidenceBadge.tsx
<Badge variant={confidence === "HIGH" ? "success" : "warning"}>
  {confidence}
</Badge>
```

### Domain Tags
```tsx
// MessageBubble.tsx
{domain && <Chip size="sm">{domain}</Chip>}
```

### Refusal Display
```tsx
// Verbatim from backend, no modification
{response.status === "refused" && (
  <Alert variant="warning">{response.refusal_reason}</Alert>
)}
```

---

## Summary

| Decision | Choice |
|----------|--------|
| Framework | Vite + React (SPA) |
| Components | shadcn/ui |
| Animations | React Bits (presentational only) |
| State | Local useState + custom hooks |
| API | fetch() with JSON |
| Security | No RSC, no SSR, escaped text only |

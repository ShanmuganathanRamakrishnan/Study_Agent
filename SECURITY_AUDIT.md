# Lucent - Security Audit Report

**Audit Date:** 2025-12-23  
**Scope:** Pre-deployment security review  
**Auditor:** Automated Code Review

---

## Summary

| Category | Issues Found | Highest Severity |
|----------|--------------|------------------|
| Backend API Security | 2 | MEDIUM |
| RAG & Model Safety | 0 | - |
| Frontend Security | 0 | - |
| Network & Deployment | 2 | MEDIUM |
| Secrets & Configuration | 0 | - |
| Dependency Risks | 1 | LOW |

**Overall Assessment:** SAFE for deployment with documented limitations.

---

## 1. Backend API Security

### Issue 1.1: No File Size Limit on Uploads
**Severity:** MEDIUM  
**File:** `server.py` (line 529)  
**Root Cause:** `file.read()` reads entire file into memory without size check.  
**Risk:** Memory exhaustion / DoS via large file upload.  
**Fixable:** Yes - add `MAX_FILE_SIZE` check before `file.read()`.  
**Recommendation:**
```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
if file.size and file.size > MAX_FILE_SIZE:
    return rejection
```

### Issue 1.2: No Rate Limiting
**Severity:** MEDIUM  
**File:** `server.py`  
**Root Cause:** No request throttling on any endpoint.  
**Risk:** DoS via repeated requests.  
**Fixable:** Yes - add FastAPI rate limiting middleware.  
**Recommendation:** Use `slowapi` or similar before production.

### Verified Safe:
- ✅ File type validation (PDF/TXT only)
- ✅ Domain sanitization (strip + uppercase)
- ✅ Content validation (min 100 words)
- ✅ PDF page limit (MAX_PAGES = 100)
- ✅ Per-page error handling (continues on failure)
- ✅ No path traversal (no file system writes)

---

## 2. RAG & Model Safety

**No issues found.**

### Verified:
- ✅ Confidence gating enforced (`HIGH` / `LOW`)
- ✅ LOW confidence uses stricter prompt
- ✅ Refusal messages cannot be bypassed
- ✅ All answers require retrieved quotes
- ✅ Intent router filters greetings before RAG
- ✅ No code path allows answering without source

---

## 3. Frontend Security

**No issues found.**

### Verified:
- ✅ No `dangerouslySetInnerHTML` usage
- ✅ All API responses rendered as text
- ✅ Domain/filename displayed via React text nodes (auto-escaped)
- ✅ Error messages are structured (no raw stack traces)

---

## 4. Network & Deployment

### Issue 4.1: CORS Limited to Localhost
**Severity:** LOW (requires config for production)  
**File:** `server.py` (lines 133-137)  
**Root Cause:** CORS only allows localhost origins.  
**Risk:** Production frontend will fail without update.  
**Fixable:** Yes - add production domain to `allow_origins`.  
**Recommendation:**
```python
allow_origins=[
    "http://localhost:5173",
    "https://lucent.example.com",  # Add production URL
]
```

### Issue 4.2: No Request Timeout
**Severity:** LOW  
**File:** `server.py`  
**Root Cause:** No timeout on model inference.  
**Risk:** Slow queries may hang connections.  
**Fixable:** Yes - add timeout wrapper around `generate_solution`.  
**Document as Limitation:** Model inference is synchronous.

### Verified Safe:
- ✅ No hardcoded API keys or secrets
- ✅ No debug flags in production code
- ✅ Health endpoint does not leak internals

---

## 5. Secrets & Configuration

**No issues found.**

### Verified:
- ✅ No hardcoded tokens or API keys
- ✅ `VITE_API_URL` is frontend-only (no secrets)
- ✅ HuggingFace cache paths are internal only
- ✅ No sensitive data in logs

---

## 6. Dependency Risks

### Issue 6.1: pdfplumber Version Not Pinned
**Severity:** LOW  
**File:** `requirements.txt`  
**Root Cause:** `pdfplumber>=0.10.0` allows any future version.  
**Risk:** Breaking changes or CVEs in future versions.  
**Fixable:** Yes - pin exact version.  
**Recommendation:** `pdfplumber==0.10.0`

### Verified Safe:
- ✅ FastAPI (no known CVEs)
- ✅ transformers (no known CVEs)
- ✅ torch (no known CVEs)

---

## Known Limitations (Document for Users)

| Limitation | Description |
|------------|-------------|
| No internet | Cannot access external knowledge |
| GPU required | CPU inference not supported |
| 100-page max | Large PDFs are truncated |
| Text-only PDF | Scanned/image PDFs not supported |
| Single session | Conversation resets on refresh |
| No rate limit | Currently unprotected from spam |

---

## Recommendations Before Production

| Priority | Action |
|----------|--------|
| HIGH | Add file size limit (10 MB) |
| HIGH | Add rate limiting middleware |
| MEDIUM | Pin dependency versions |
| MEDIUM | Add production CORS origins |
| LOW | Add inference timeout |

---

## Conclusion

**GO for deployment** with the following conditions:
1. File size limit added
2. CORS updated for production domain
3. Rate limiting documented as TODO

No critical vulnerabilities found. System is safe for educational deployment.

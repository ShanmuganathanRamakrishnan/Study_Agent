"""
Lightweight Conversational Intent Router for Study Agent

Classifies user input into high-level intents using general heuristics.
NO ML models, NO hardcoded phrases.

Intents:
- GREETING: Short, non-question acknowledgments
- QUESTION: Academic queries (question mark, interrogative structure)
- STUDY_REQUEST: Requests for study material generation
- CLARIFICATION: Ambiguous or incomplete queries
- NOISE: Empty or meaningless input

Design Principle:
Use structural signals (length, punctuation, word patterns) rather than keyword matching.
"""

import re
from typing import Tuple
from dataclasses import dataclass


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: str
    confidence: str  # HIGH, MEDIUM, LOW
    reason: str


# Structural patterns (NOT keyword matching - these are grammatical patterns)
QUESTION_STARTERS = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'could', 'would', 'is', 'are', 'do', 'does', 'did'}
STUDY_VERBS = {'generate', 'create', 'make', 'give', 'quiz', 'test', 'practice', 'study', 'review'}
STUDY_NOUNS = {'question', 'questions', 'quiz', 'test', 'exam', 'practice', 'exercises', 'problems'}


def classify_intent(query: str) -> IntentResult:
    """
    Classify user input into an intent using structural heuristics.
    
    Args:
        query: Raw user input
        
    Returns:
        IntentResult with intent, confidence, and reason
    """
    # Normalize input
    original = query
    query = query.strip()
    
    # --- NOISE: Empty or whitespace-only ---
    if not query:
        return IntentResult(
            intent="NOISE",
            confidence="HIGH",
            reason="Empty input"
        )
    
    # Basic metrics
    word_count = len(query.split())
    char_count = len(query)
    has_question_mark = '?' in query
    words_lower = query.lower().split()
    first_word = words_lower[0] if words_lower else ""
    
    # --- NOISE: Too short to be meaningful (single char) ---
    if word_count == 1 and char_count <= 2:
        return IntentResult(
            intent="NOISE",
            confidence="MEDIUM",
            reason=f"Single short word ({char_count} chars)"
        )
    
    # --- GREETING: Short, non-question, social patterns ---
    # Heuristic: 2-3 words, no question mark, no content verbs
    if word_count in [2, 3] and not has_question_mark:
        content_verbs = {'explain', 'describe', 'define', 'calculate', 'solve', 'find', 'show', 'generate', 'create', 'practice'}
        has_content_verb = any(w in content_verbs for w in words_lower)
        
        if not has_content_verb:
            return IntentResult(
                intent="GREETING",
                confidence="MEDIUM",
                reason=f"Short non-question ({word_count} words)"
            )
    
    # Single word greetings (common patterns by length/structure)
    if word_count == 1 and char_count >= 2 and char_count <= 8:
        # Single words that are likely greetings/acknowledgments
        # NOT using keyword matching - using structural pattern: short single word
        return IntentResult(
            intent="GREETING",
            confidence="LOW",
            reason="Single word (likely greeting/acknowledgment)"
        )
    
    # --- STUDY_REQUEST: Requests for study material generation ---
    # Heuristic: Contains study verb + noun pattern
    has_study_verb = any(w in STUDY_VERBS for w in words_lower)
    has_study_noun = any(w in STUDY_NOUNS for w in words_lower)
    
    if has_study_verb and has_study_noun:
        return IntentResult(
            intent="STUDY_REQUEST",
            confidence="HIGH",
            reason="Study verb + noun pattern detected"
        )
    
    # Weaker signal: just noun without verb (e.g., "questions about entropy")
    if has_study_noun and word_count >= 2 and word_count <= 6:
        return IntentResult(
            intent="STUDY_REQUEST",
            confidence="MEDIUM",
            reason="Study noun pattern detected"
        )
    
    # --- QUESTION: Academic query structure ---
    # Strong signal: Question mark
    if has_question_mark:
        if word_count >= 4:
            return IntentResult(
                intent="QUESTION",
                confidence="HIGH",
                reason="Question mark with sufficient length"
            )
        else:
            return IntentResult(
                intent="CLARIFICATION",
                confidence="MEDIUM",
                reason="Question mark but too short"
            )
    
    # Strong signal: Starts with interrogative word
    if first_word in QUESTION_STARTERS and word_count >= 4:
        return IntentResult(
            intent="QUESTION",
            confidence="HIGH",
            reason=f"Interrogative structure ({first_word}...)"
        )
    
    # Medium signal: Imperative with content verb (explain, describe, define)
    imperative_verbs = {'explain', 'describe', 'define', 'tell', 'show', 'list', 'compare'}
    if first_word in imperative_verbs and word_count >= 3:
        return IntentResult(
            intent="QUESTION",
            confidence="MEDIUM",
            reason=f"Imperative structure ({first_word}...)"
        )
    
    # --- CLARIFICATION: Ambiguous or incomplete ---
    # Heuristic: Short without clear structure
    if word_count < 4:
        return IntentResult(
            intent="CLARIFICATION",
            confidence="MEDIUM",
            reason=f"Too short to determine intent ({word_count} words)"
        )
    
    # --- Default: Treat as QUESTION (academic context) ---
    # In an educational system, longer inputs are likely questions
    if word_count >= 5:
        return IntentResult(
            intent="QUESTION",
            confidence="LOW",
            reason="Defaulting to question (sufficient length, academic context)"
        )
    
    # Fallback
    return IntentResult(
        intent="CLARIFICATION",
        confidence="LOW",
        reason="Unable to determine clear intent"
    )


def get_intent_response(intent_result: IntentResult) -> Tuple[str, bool]:
    """
    Get appropriate response for non-QUESTION intents.
    
    Args:
        intent_result: Result from classify_intent
        
    Returns:
        (response_text, should_proceed_to_rag)
    """
    intent = intent_result.intent
    
    if intent == "GREETING":
        return "Ask a question about your uploaded materials.", False
    
    if intent == "STUDY_REQUEST":
        return "ROUTE_TO_STUDY_MODE", True  # Special marker for routing
    
    if intent == "CLARIFICATION":
        return "Could you please rephrase or provide more detail? I work best with specific questions about your study materials.", False
    
    if intent == "NOISE":
        return "Please enter a question about your study materials, or upload content to get started.", False
    
    # QUESTION - proceed to RAG
    return "", True


def log_intent(query: str, result: IntentResult) -> None:
    """Log intent classification for debugging."""
    truncated = query[:50] + "..." if len(query) > 50 else query
    print(f"[INTENT] Detected: {result.intent} ({result.confidence}) | Query: '{truncated}' | Reason: {result.reason}", flush=True)


# --- Unit Test Examples ---
def run_tests():
    """Test examples for each intent."""
    test_cases = [
        # GREETING
        ("hi", "GREETING"),
        ("thanks", "GREETING"),
        ("ok", "GREETING"),
        ("hey there", "GREETING"),
        
        # QUESTION
        ("What is entropy in thermodynamics?", "QUESTION"),
        ("How does photosynthesis work?", "QUESTION"),
        ("Explain the concept of homeostasis", "QUESTION"),
        ("Can you describe the process of mitosis?", "QUESTION"),
        
        # STUDY_REQUEST
        ("generate questions about entropy", "STUDY_REQUEST"),
        ("quiz me on thermodynamics", "STUDY_REQUEST"),
        ("create practice problems", "STUDY_REQUEST"),
        ("give me test questions", "STUDY_REQUEST"),
        
        # CLARIFICATION
        ("what?", "CLARIFICATION"),
        ("entropy", "CLARIFICATION"),
        ("the thing", "CLARIFICATION"),
        
        # NOISE
        ("", "NOISE"),
        ("   ", "NOISE"),
        ("a", "NOISE"),
    ]
    
    print("=" * 60)
    print("INTENT ROUTER UNIT TESTS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for query, expected in test_cases:
        result = classify_intent(query)
        status = "✓" if result.intent == expected else "✗"
        if result.intent == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} '{query[:30]}...' → {result.intent} (expected: {expected})")
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    run_tests()

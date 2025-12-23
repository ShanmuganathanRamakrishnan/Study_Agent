"""
Conversation Context Buffer for Study Agent

Maintains a bounded window of recent conversation turns for:
1. Follow-up question resolution
2. Context-aware generation

Design Principles:
- Buffer used ONLY for generation context, NOT for indexing
- Retrieval confidence still gates answers (no hallucination)
- Greetings handled separately by intent router
- Generic follow-up detection (no topic hardcoding)
"""

from typing import List, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    domain: Optional[str] = None


class ConversationBuffer:
    """
    Bounded buffer storing last 2 user messages for context.
    Minimal memory - just enough for follow-up resolution.
    """
    
    def __init__(self, max_user_turns: int = 2):
        self.max_user_turns = max_user_turns
        self.user_queries: deque = deque(maxlen=max_user_turns)
        self.last_assistant_response: Optional[str] = None
        self.last_domain: Optional[str] = None
    
    def add_user_query(self, query: str) -> None:
        """Add a user query to the buffer."""
        self.user_queries.append(query)
    
    def add_assistant_response(self, response: str, domain: Optional[str] = None) -> None:
        """Store the last assistant response and domain."""
        self.last_assistant_response = response
        self.last_domain = domain
    
    def get_previous_user_query(self) -> Optional[str]:
        """Get the PREVIOUS user query (not the current one)."""
        if len(self.user_queries) >= 2:
            return self.user_queries[-2]
        return None
    
    def get_last_domain(self) -> Optional[str]:
        """Get the domain from the most recent response."""
        return self.last_domain
    
    def get_recent_context(self) -> str:
        """Get formatted recent user queries for context."""
        if len(self.user_queries) == 0:
            return ""
        
        lines = ["[Recent User Context]"]
        for i, query in enumerate(self.user_queries):
            truncated = query[:150] + "..." if len(query) > 150 else query
            lines.append(f"- {truncated}")
        
        return "\n".join(lines)
    
    def has_context(self) -> bool:
        """Check if there's prior context available."""
        return len(self.user_queries) >= 1 and self.last_domain is not None
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.user_queries.clear()
        self.last_assistant_response = None
        self.last_domain = None


def is_follow_up_query(query: str, buffer: ConversationBuffer) -> bool:
    """
    Detect if a query is a follow-up referencing prior context.
    
    Heuristics (NO topic hardcoding):
    - Short query (< 6 words)
    - Contains referential words
    - Has prior conversation context
    """
    if not buffer.has_context():
        return False
    
    words = query.lower().split()
    word_count = len(words)
    
    # Very short queries with context are likely follow-ups
    if word_count <= 5:
        return True
    
    # Referential patterns
    referential_words = {'that', 'it', 'this', 'those', 'these', 'the', 'them', 'more', 'also'}
    has_referential = any(w in referential_words for w in words)
    
    if has_referential and word_count < 10:
        return True
    
    return False


def expand_follow_up_query(query: str, buffer: ConversationBuffer) -> str:
    """
    Expand a follow-up query with key terms from previous conversation.
    
    Uses:
    - Key terms from previous user query
    - Domain from last response
    """
    prev_query = buffer.get_previous_user_query()
    last_domain = buffer.get_last_domain()
    
    # Extract key terms from previous query
    key_terms = []
    if prev_query:
        # Get longer words (likely topic words) - NOT hardcoded topics
        words = prev_query.lower().split()
        key_terms = [w for w in words if len(w) >= 3 and w.isalpha()][:5]
    
    # If no previous query, try using last user query in buffer (current turn's predecessor)
    if not key_terms and len(buffer.user_queries) >= 1:
        last_q = list(buffer.user_queries)[-1] if buffer.user_queries else ""
        words = last_q.lower().split()
        key_terms = [w for w in words if len(w) >= 3 and w.isalpha()][:5]
    
    # Build expanded query
    if key_terms:
        expanded = f"{query} {' '.join(key_terms)}"
    else:
        expanded = query
    
    print(f"[CONTEXT] Expanded: '{query}' -> '{expanded}'", flush=True)
    return expanded


# Global buffer (simplified)
_buffer = ConversationBuffer(max_user_turns=2)


def get_buffer() -> ConversationBuffer:
    """Get the global conversation buffer."""
    return _buffer


def reset_buffer() -> None:
    """Reset the conversation buffer."""
    _buffer.clear()
    print("[CONTEXT] Buffer cleared", flush=True)

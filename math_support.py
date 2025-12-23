"""
Bounded Math Support Module for Study Agent

Provides VERY LIMITED math support while preserving zero-hallucination guarantees.

ALLOWED (Simple Math):
- Single-step arithmetic: 5 + 3, 12 * 4, 100 / 5, 15 - 7
- Simple percentages: 20% of 100, what is 15% of 80
- Basic unit conversions mentioned in uploaded material
- Application of formulas FROM uploaded material (e.g., "Calculate entropy using S = k ln Ω")

REFUSED (Complex Math):
- Multi-step algebra (solve for x in 3x + 5 = 2x - 1)
- Equations with multiple variables
- Calculus (derivatives, integrals, limits)
- Linear algebra (matrices, vectors, eigenvalues)
- Proofs or symbolic manipulation
- Statistics beyond simple mean/percentage
- Physics/engineering calculations beyond simple substitution
- Any ambiguous or underspecified problems

DESIGN PRINCIPLE:
If in doubt, REFUSE. Better to say "I cannot solve complex math" than to hallucinate.
"""

import re
from typing import Tuple, Optional


# Patterns that indicate COMPLEX math (should refuse)
COMPLEX_MATH_PATTERNS = [
    # Calculus
    r'\b(derivative|differentiate|d/dx|integral|integrate|∫|limit|lim)\b',
    r'\b(dy/dx|∂|partial)\b',
    
    # Linear algebra
    r'\b(matrix|matrices|vector|eigenvalue|eigenvector|determinant|transpose)\b',
    r'\b(dot product|cross product|linear transformation)\b',
    
    # Proofs and abstract
    r'\b(prove|proof|theorem|lemma|QED|∀|∃|therefore)\b',
    r'\b(induction|contradiction|contrapositive)\b',
    
    # Multi-variable
    r'\b(system of equations|simultaneous)\b',
    r'[a-z]\s*[,]\s*[a-z]\s*[,]\s*[a-z]',  # x, y, z pattern
    
    # Complex expressions
    r'\b(factor|factorize|expand|simplify)\b',
    r'\b(polynomial|quadratic|cubic|quartic)\b',
    r'\b(logarithm|log|ln|exponential)\s*equation',
    
    # Statistics beyond basics
    r'\b(standard deviation|variance|regression|correlation|hypothesis test)\b',
    r'\b(chi-?square|t-?test|p-?value|confidence interval)\b',
    
    # Trigonometry beyond basics
    r'\b(sin|cos|tan|cot|sec|csc)\s*\^?\s*-?1?\s*\(',
    
    # Complex notation
    r'∑|∏|√|∛|∜',
    r'\bsum\s+from\b',
]

# Patterns that indicate SIMPLE math (may allow)
SIMPLE_MATH_PATTERNS = [
    # Basic arithmetic
    r'^\s*\d+\s*[\+\-\*\/\×\÷]\s*\d+\s*$',
    r'\bwhat\s+is\s+\d+\s*[\+\-\*\/\×\÷]\s*\d+\b',
    
    # Simple percentages
    r'\d+\s*%\s*of\s*\d+',
    r'what\s+is\s+\d+\s*%',
    r'calculate\s+\d+\s*%',
    
    # Simple formula application
    r'\bcalculate\s+using\b',
    r'\bsubstitute\s+into\b',
    r'\bplug\s+in\b',
]


def is_simple_arithmetic(query: str) -> Tuple[bool, Optional[str]]:
    """
    Check if query is simple arithmetic that can be safely computed.
    
    Returns:
        (is_simple, expression) where expression is the arithmetic to compute if simple
    """
    query_lower = query.lower().strip()
    
    # Pattern: "what is X + Y" or just "X + Y"
    arithmetic_pattern = r'(?:what\s+is\s+)?(\d+(?:\.\d+)?)\s*([\+\-\*\/\×\÷])\s*(\d+(?:\.\d+)?)'
    match = re.search(arithmetic_pattern, query_lower)
    
    if match:
        num1 = float(match.group(1))
        op = match.group(2)
        num2 = float(match.group(3))
        
        # Standardize operators
        if op in ['×', '*']:
            op = '*'
        elif op in ['÷', '/']:
            op = '/'
        
        return True, f"{num1} {op} {num2}"
    
    return False, None


def compute_simple_arithmetic(expression: str) -> Optional[float]:
    """
    Safely compute a simple arithmetic expression.
    Returns None if computation fails.
    """
    try:
        # Parse the expression
        match = re.match(r'(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)', expression)
        if not match:
            return None
        
        num1 = float(match.group(1))
        op = match.group(2)
        num2 = float(match.group(3))
        
        if op == '+':
            return num1 + num2
        elif op == '-':
            return num1 - num2
        elif op == '*':
            return num1 * num2
        elif op == '/':
            if num2 == 0:
                return None  # Division by zero
            return num1 / num2
        
        return None
    except:
        return None


def is_simple_percentage(query: str) -> Tuple[bool, Optional[Tuple[float, float]]]:
    """
    Check if query is a simple percentage calculation.
    
    Returns:
        (is_percentage, (percentage, base)) if simple percentage
    """
    query_lower = query.lower().strip()
    
    # Pattern: "X% of Y" or "what is X% of Y"
    percentage_pattern = r'(?:what\s+is\s+)?(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)'
    match = re.search(percentage_pattern, query_lower)
    
    if match:
        percentage = float(match.group(1))
        base = float(match.group(2))
        return True, (percentage, base)
    
    return False, None


def compute_percentage(percentage: float, base: float) -> float:
    """Compute X% of Y."""
    return (percentage / 100) * base


def classify_math_query(query: str) -> Tuple[str, Optional[str]]:
    """
    Classify a math-like query.
    
    Returns:
        (classification, reason)
        
        classification: "SIMPLE_ARITHMETIC" | "SIMPLE_PERCENTAGE" | "COMPLEX_REFUSED" | "NOT_MATH"
    """
    query_lower = query.lower()
    
    # First, check for complex math indicators (refuse these)
    for pattern in COMPLEX_MATH_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return "COMPLEX_REFUSED", f"Complex math detected (pattern: {pattern[:30]}...)"
    
    # Check for simple arithmetic
    is_simple, expr = is_simple_arithmetic(query)
    if is_simple:
        return "SIMPLE_ARITHMETIC", expr
    
    # Check for simple percentage
    is_pct, pct_data = is_simple_percentage(query)
    if is_pct:
        return "SIMPLE_PERCENTAGE", f"{pct_data[0]}% of {pct_data[1]}"
    
    # Check for other simple math patterns
    for pattern in SIMPLE_MATH_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            # Has simple math indicators but didn't match specific patterns
            # This might be formula application - let RAG handle it
            return "NOT_MATH", None
    
    return "NOT_MATH", None


def solve_simple_math(query: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Attempt to solve a simple math problem.
    
    Returns:
        (solved, answer, explanation)
        
        If solved=False, answer contains refusal reason.
    """
    classification, data = classify_math_query(query)
    
    if classification == "COMPLEX_REFUSED":
        return False, None, "This math problem exceeds the scope of simple arithmetic. I can only help with basic single-step calculations like addition, subtraction, multiplication, division, and simple percentages."
    
    if classification == "SIMPLE_ARITHMETIC" and data:
        result = compute_simple_arithmetic(data)
        if result is not None:
            # Format result nicely
            if result == int(result):
                result_str = str(int(result))
            else:
                result_str = f"{result:.4f}".rstrip('0').rstrip('.')
            
            return True, result_str, f"Calculated: {data} = {result_str}"
    
    if classification == "SIMPLE_PERCENTAGE" and data:
        is_pct, pct_data = is_simple_percentage(query)
        if is_pct and pct_data:
            result = compute_percentage(pct_data[0], pct_data[1])
            if result == int(result):
                result_str = str(int(result))
            else:
                result_str = f"{result:.4f}".rstrip('0').rstrip('.')
            
            return True, result_str, f"Calculated: {pct_data[0]}% of {pct_data[1]} = {result_str}"
    
    # Not a math problem or not solvable - let RAG handle it
    return False, None, None


def should_attempt_math(query: str) -> bool:
    """
    Quick check if query looks like it might be a math problem.
    """
    query_lower = query.lower()
    
    # Math keywords
    math_keywords = [
        'calculate', 'compute', 'solve', 'what is', 'find',
        '+', '-', '*', '/', '×', '÷', '%', '=',
        'plus', 'minus', 'times', 'divided', 'percent'
    ]
    
    return any(kw in query_lower for kw in math_keywords)

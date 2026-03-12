"""
src/agents/calculator_agent.py
────────────────────────────────
Specialized sub-agent for numerical and arithmetic reasoning.
Safely evaluates mathematical expressions so the LLM doesn't hallucinate numbers.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Union

from langchain_core.tools import Tool
from loguru import logger


# Allowed operators and functions for safe evaluation
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_NAMES = {
    "abs": abs, "round": round,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "exp": math.exp, "floor": math.floor, "ceil": math.ceil,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e,
    "min": min, "max": max, "sum": sum,
}


def _safe_eval(node: ast.AST) -> Union[int, float]:
    """Recursively evaluate an AST node with only safe operations."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.Name):
        if node.id in _SAFE_NAMES:
            return _SAFE_NAMES[node.id]
        raise NameError(f"Name '{node.id}' is not allowed.")
    elif isinstance(node, ast.BinOp):
        op = _OPERATORS.get(type(node.op))
        if op is None:
            raise TypeError(f"Unsupported operator: {type(node.op)}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return op(left, right)
    elif isinstance(node, ast.UnaryOp):
        op = _OPERATORS.get(type(node.op))
        if op is None:
            raise TypeError(f"Unsupported unary operator: {type(node.op)}")
        return op(_safe_eval(node.operand))
    elif isinstance(node, ast.Call):
        func = _safe_eval(node.func)
        args = [_safe_eval(a) for a in node.args]
        return func(*args)
    elif isinstance(node, ast.Attribute):
        # e.g., math.sqrt — not needed but handled defensively
        raise ValueError("Attribute access not allowed.")
    else:
        raise TypeError(f"Unsupported AST node type: {type(node)}")


def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression string.

    Supports: +, -, *, /, **, %, sqrt, log, exp, floor, ceil, sin, cos, tan, pi, e, abs, round, min, max, sum.

    Args:
        expression: mathematical expression as a string, e.g. "sqrt(144) + 2**3"

    Returns:
        Result as a string, or an error message.
    """
    expression = expression.strip()
    logger.debug(f"Calculator: evaluating '{expression}'")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)

        # Round floats to 8 significant figures to avoid floating point noise
        if isinstance(result, float):
            result = round(result, 8)

        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except (NameError, TypeError, ValueError) as e:
        return f"Error: {e}"
    except SyntaxError:
        return f"Error: Invalid expression syntax — '{expression}'"
    except Exception as e:
        return f"Error: {e}"


class CalculatorAgent:
    """Wraps the safe calculator as a LangChain Tool."""

    @staticmethod
    def as_tool() -> Tool:
        return Tool(
            name="calculator",
            func=calculate,
            description=(
                "Useful for arithmetic, algebra, and numerical reasoning. "
                "Input: a mathematical expression string such as '(100 * 1.23) / 4 + sqrt(16)'. "
                "Supported: +, -, *, /, **, %, sqrt, log, exp, floor, ceil, sin, cos, tan, pi, e, abs, round, min, max. "
                "Output: the computed numeric result."
            ),
        )

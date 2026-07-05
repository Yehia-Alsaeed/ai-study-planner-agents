"""Tool helpers used by CrewAI agents."""

from __future__ import annotations

import ast
import operator
from typing import Any

from study_planner.difficulty import SubjectDifficultyClassifier


class UnsafeExpressionError(ValueError):
    """Raised when calculator input contains unsupported syntax."""


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_calculate(expression: str) -> str:
    """Evaluate simple arithmetic without using ``eval``."""

    try:
        parsed = ast.parse(expression, mode="eval")
        value = _evaluate_node(parsed.body)
    except UnsafeExpressionError:
        raise
    except Exception as exc:
        raise UnsafeExpressionError(f"Invalid arithmetic expression: {expression}") from exc

    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def build_crewai_tools(
    *,
    classifier: SubjectDifficultyClassifier | None = None,
    search_results: int = 3,
) -> dict[str, Any]:
    """Build CrewAI-compatible tools with imports delayed until runtime."""

    from crewai.tools import tool
    from crewai_tools import SerperDevTool

    classifier = classifier or SubjectDifficultyClassifier()
    search_tool = SerperDevTool(n_results=search_results)

    @tool("Calculator")
    def calculator_tool(expression: str) -> str:
        """Evaluates a mathematical expression and returns the result as a string."""

        try:
            return safe_calculate(expression)
        except UnsafeExpressionError as exc:
            return f"Error: {exc}"

    @tool("SubjectDifficultyClassifier")
    def difficulty_classifier_tool(subject_name: str) -> str:
        """Classifies a subject as hard, medium, or easy using semantic similarity."""

        return classifier.classify(subject_name).to_tool_message()

    return {
        "search": search_tool,
        "calculator": calculator_tool,
        "difficulty_classifier": difficulty_classifier_tool,
    }


def _evaluate_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand))
    raise UnsafeExpressionError(f"Unsupported calculator expression: {ast.dump(node)}")

"""CrewAI agent factory functions."""

from __future__ import annotations

from typing import Any


def build_llm(*, model: str = "gpt-4o-mini", temperature: float = 0.2, max_retries: int = 3):
    from crewai import LLM

    return LLM(model=model, temperature=temperature, max_retries=max_retries)


def build_agents(*, llm: Any, tools: dict[str, Any]) -> dict[str, Any]:
    from crewai import Agent

    profiler_agent = Agent(
        role="Student Profiler",
        goal=(
            "Extract and structure all student information including subjects, difficulty levels, "
            "exam dates, and available study hours per day. Use the SubjectDifficultyClassifier "
            "tool to classify every subject before building the profile."
        ),
        backstory=(
            "An expert academic advisor who uses semantic tools to assess subject difficulty "
            "before building a student profile."
        ),
        tools=[tools["difficulty_classifier"]],
        llm=llm,
        verbose=True,
    )
    generator_agent = Agent(
        role="Study Plan Generator",
        goal=(
            "Generate a realistic day-by-day study plan based on the student profile, using "
            "search for study strategies and calculator support for hour distribution."
        ),
        backstory=(
            "An experienced academic coach who builds balanced schedules tailored to student "
            "capacity and exam deadlines."
        ),
        tools=[tools["search"], tools["calculator"]],
        llm=llm,
        verbose=True,
    )
    critic_agent = Agent(
        role="Plan Critic",
        goal=(
            "Evaluate study plans against constraints, identify overloaded days, missing buffer "
            "days, neglected subjects, and incorrect exam handling, then return a quality score."
        ),
        backstory="A strict academic reviewer who checks plans for practical and logical flaws.",
        tools=[tools["calculator"]],
        llm=llm,
        verbose=True,
    )
    optimizer_agent = Agent(
        role="Plan Optimizer",
        goal=(
            "Use the original plan and critic feedback to produce a corrected final schedule "
            "with clear explanations for changes."
        ),
        backstory="A senior academic strategist who converts critique into practical schedules.",
        tools=[tools["search"], tools["calculator"]],
        llm=llm,
        verbose=True,
    )
    return {
        "profiler": profiler_agent,
        "generator": generator_agent,
        "critic": critic_agent,
        "optimizer": optimizer_agent,
    }

"""CrewAI task definitions for the study planner workflow."""

from __future__ import annotations

from typing import Any


def build_tasks(*, agents: dict[str, Any]) -> list[Any]:
    from crewai import Task

    profiling_task = Task(
        description=(
            "Take the following raw student input and extract all relevant information. "
            "For EVERY subject mentioned, call the SubjectDifficultyClassifier tool to determine "
            "its difficulty. Produce a structured profile with subjects, difficulty levels, exam "
            "days, total study days counted inclusively, and daily available study hours. "
            "Raw input: {student_input}"
        ),
        expected_output=(
            "A structured student profile with each subject's difficulty level, exam day, "
            "total study days, and daily available study hours."
        ),
        agent=agents["profiler"],
    )
    generation_task = Task(
        description=(
            "Follow these steps before writing the schedule.\n"
            "1. For every subject, use search for 'best study strategies for [subject name] "
            "computer science university exam'.\n"
            "2. Use the calculator for total hours and difficulty pool calculations.\n"
            "3. Write a locked day table where each buffer day is exam_day - 1 and each exam "
            "day is the exact exam day.\n"
            "4. Fill every remaining day from Day 1 to the last exam day. No day may exceed "
            "the student's daily available hours. Never schedule new material for a subject "
            "on its buffer day, exam day, or after its exam day.\n"
            "Student profile: {student_input}"
        ),
        expected_output=(
            "A locked day table followed by a full day-by-day study plan. Every buffer day and "
            "exam day must be clearly labeled."
        ),
        agent=agents["generator"],
    )
    critique_task = Task(
        description=(
            "Review the generated study plan against the student profile. Check hour limits, "
            "post-exam scheduling, buffer days, day coverage, hard-subject prioritization, and "
            "correct exam-day labels. End with exactly: Quality Score: X/10"
        ),
        expected_output=(
            "A numbered list of every issue found with specific day numbers and subject names. "
            "End with exactly: Quality Score: X/10"
        ),
        agent=agents["critic"],
    )
    optimization_task = Task(
        description=(
            "Using the original study plan and every issue the critic identified, produce a "
            "corrected final schedule. First write a verification table showing each subject's "
            "exam day and buffer day. Then write the final day-by-day plan. Enforce correct "
            "buffer labels, exam labels, no post-exam study, and the daily hour limit. "
            "End with exactly: Quality Score: X/10"
        ),
        expected_output=(
            "A verification table, the final optimized day-by-day plan, and a bullet list of "
            "changes made. End with exactly: Quality Score: X/10"
        ),
        agent=agents["optimizer"],
    )
    return [profiling_task, generation_task, critique_task, optimization_task]

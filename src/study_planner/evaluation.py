"""Structured evaluation helpers for study planner outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


HOUR_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*hours?", re.IGNORECASE)
DAY_PATTERN = re.compile(r"\bDay\s*(\d+)\b\s*:?\s*(.*)", re.IGNORECASE)
QUALITY_PATTERN = re.compile(r"Quality Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", re.IGNORECASE)


@dataclass(frozen=True)
class SubjectExam:
    name: str
    exam_day: int
    difficulty: str | None = None

    def to_json_dict(self) -> dict[str, str | int | None]:
        return {
            "name": self.name,
            "exam_day": self.exam_day,
            "difficulty": self.difficulty,
        }


@dataclass(frozen=True)
class StudentProfile:
    subjects: list[SubjectExam]
    daily_available_hours: float
    start_day: int = 1

    @property
    def subject_names(self) -> list[str]:
        return [subject.name for subject in self.subjects]

    @property
    def last_exam_day(self) -> int:
        return max(subject.exam_day for subject in self.subjects)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "start_day": self.start_day,
            "daily_available_hours": self.daily_available_hours,
            "last_exam_day": self.last_exam_day,
            "subjects": [subject.to_json_dict() for subject in self.subjects],
        }


@dataclass(frozen=True)
class ParsedDay:
    day: int
    text: str
    total_hours: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "day": self.day,
            "text": self.text,
            "total_hours": self.total_hours,
        }


@dataclass(frozen=True)
class PlanEvaluation:
    scores: dict[str, float]
    parsed_days: list[ParsedDay]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "scores": self.scores,
            "parsed_days": [day.to_json_dict() for day in self.parsed_days],
        }


@dataclass(frozen=True)
class PlannerRunRecord:
    variation_name: str
    temperature: float
    model: str
    profile: StudentProfile
    raw_result: str
    evaluation: PlanEvaluation

    def to_json_dict(self) -> dict[str, object]:
        return {
            "variation_name": self.variation_name,
            "temperature": self.temperature,
            "model": self.model,
            "profile": self.profile.to_json_dict(),
            "raw_result": self.raw_result,
            "evaluation": self.evaluation.to_json_dict(),
        }

    def write_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_json_dict(), indent=2),
            encoding="utf-8",
        )
        return output_path


def sample_student_profile() -> StudentProfile:
    return StudentProfile(
        subjects=[
            SubjectExam(name="NLP", exam_day=14, difficulty="hard"),
            SubjectExam(name="Deep Learning", exam_day=12, difficulty="hard"),
            SubjectExam(name="Security", exam_day=10, difficulty="medium"),
            SubjectExam(name="Database", exam_day=8, difficulty="medium"),
        ],
        daily_available_hours=4,
        start_day=1,
    )


def sample_student_input() -> str:
    return (
        "I have 4 subjects: NLP (hard), Deep Learning (hard), Security (medium), "
        "and Database (medium). My exams are: NLP on day 14, Deep Learning on day 12, "
        "Security on day 10, and Database on day 8. I have 4 hours available to study "
        "every day starting from today which is day 1."
    )


def parse_plan_days(result: str) -> list[ParsedDay]:
    parsed_days: list[ParsedDay] = []
    for line in str(result).splitlines():
        match = DAY_PATTERN.search(line)
        if not match:
            continue
        hours = [float(value) for value in HOUR_PATTERN.findall(line)]
        parsed_days.append(
            ParsedDay(
                day=int(match.group(1)),
                text=line.strip(),
                total_hours=sum(hours),
            )
        )
    return parsed_days


def plan_completeness_score(result: str, subjects: list[str]) -> float:
    if not subjects:
        return 0.0
    result_text = str(result).lower()
    matches = sum(1 for subject in subjects if subject.lower() in result_text)
    return matches / len(subjects)


def constraint_satisfaction_score(parsed_days: list[ParsedDay], daily_hour_limit: float) -> float:
    days_with_hours = [day for day in parsed_days if day.total_hours > 0]
    if not days_with_hours:
        return 0.0
    valid_days = sum(1 for day in days_with_hours if day.total_hours <= daily_hour_limit)
    return valid_days / len(days_with_hours)


def quality_score(result: str) -> float:
    match = QUALITY_PATTERN.search(str(result))
    if not match:
        return 0.0
    return float(match.group(1)) / 10


def evaluate_plan(result: str, profile: StudentProfile) -> PlanEvaluation:
    parsed_days = parse_plan_days(result)
    scores = {
        "completeness": plan_completeness_score(result, profile.subject_names),
        "constraint_satisfaction": constraint_satisfaction_score(
            parsed_days,
            profile.daily_available_hours,
        ),
        "quality": quality_score(result),
    }
    return PlanEvaluation(scores=scores, parsed_days=parsed_days)


def create_run_record(
    *,
    variation_name: str,
    temperature: float,
    model: str,
    raw_result: str,
    profile: StudentProfile,
) -> PlannerRunRecord:
    return PlannerRunRecord(
        variation_name=variation_name,
        temperature=temperature,
        model=model,
        profile=profile,
        raw_result=str(raw_result),
        evaluation=evaluate_plan(str(raw_result), profile),
    )

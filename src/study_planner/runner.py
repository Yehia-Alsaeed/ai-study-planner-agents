"""High-level execution helpers for notebooks and scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from study_planner.agents import build_agents, build_llm
from study_planner.config import ProjectPaths
from study_planner.evaluation import (
    StudentProfile,
    create_run_record,
    sample_student_input,
    sample_student_profile,
)
from study_planner.tasks import build_tasks
from study_planner.tools import build_crewai_tools


@dataclass(frozen=True)
class VariationConfig:
    name: str
    temperature: float
    student_input: str = field(default_factory=sample_student_input)
    profile: StudentProfile = field(default_factory=sample_student_profile)
    model: str = "gpt-4o-mini"


DEFAULT_VARIATIONS = [
    VariationConfig(name="temperature_0_2", temperature=0.2),
    VariationConfig(name="temperature_0_6", temperature=0.6),
    VariationConfig(name="temperature_0_8", temperature=0.8),
]


def run_variation(config: VariationConfig):
    from crewai import Crew, Process

    llm = build_llm(model=config.model, temperature=config.temperature)
    tools = build_crewai_tools()
    agents = build_agents(llm=llm, tools=tools)
    tasks = build_tasks(agents=agents)
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )
    raw_result = crew.kickoff(inputs={"student_input": config.student_input})
    return create_run_record(
        variation_name=config.name,
        temperature=config.temperature,
        model=config.model,
        raw_result=str(raw_result),
        profile=config.profile,
    )


def save_run_artifacts(record, paths: ProjectPaths) -> dict[str, Path]:
    text_path = paths.results_dir / f"{record.variation_name}.txt"
    json_path = paths.results_dir / f"{record.variation_name}.json"
    text_path.write_text(record.raw_result, encoding="utf-8")
    record.write_json(json_path)
    return {"text": text_path, "json": json_path}

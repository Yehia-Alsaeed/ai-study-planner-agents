# AI Study Planner Agents

AI Study Planner Agents is a reusable Python project and demo notebook for generating personalized study schedules from subjects, exam dates, and available daily study hours.

The project combines CrewAI agents, GPT-4o mini, sentence-transformer subject difficulty classification, tool-augmented planning, safe arithmetic helpers, and structured evaluation artifacts. The notebook is now a thin demo over the `study_planner` package, so the core logic can be tested, reused, and extended without copying notebook cells.

## Key Features

- **Multi-agent planning workflow:** Uses specialized agents for student profiling, study plan generation, plan critique, and plan optimization.
- **Transformer-based difficulty classification:** Uses `all-MiniLM-L6-v2` sentence embeddings to classify subjects as hard, medium, or easy through semantic similarity.
- **Tool-augmented reasoning:** Adds external search, safe calculator support, and a custom subject difficulty classifier.
- **Configurable output paths:** Defaults to local `outputs/` artifacts, with optional Google Colab Drive support through `configure_colab_paths(...)`.
- **Environment loading:** Reads API keys from `.env` first, then prompts only for missing keys.
- **Structured evaluation:** Writes raw text and structured JSON per run, including parsed day entries and scores derived from the student profile.

## System Workflow

1. The student provides subjects, exam dates, and daily available study hours.
2. The transformer embedding model classifies the difficulty of each subject.
3. The Student Profiler converts the raw request into structured planning data.
4. The Study Plan Generator creates a day-by-day schedule.
5. The Plan Critic checks the schedule for overloaded days, missing buffer days, incorrect exam handling, and weak subject prioritization.
6. The Plan Optimizer revises the schedule using the critic feedback.
7. The evaluation logic parses the final plan, scores it, and saves structured JSON artifacts.

## Agent Architecture

| Agent | Responsibility |
| --- | --- |
| Student Profiler | Extracts subjects, exam dates, study capacity, and subject difficulty. |
| Study Plan Generator | Builds an initial day-by-day plan using planning constraints and tool support. |
| Plan Critic | Reviews the generated plan for logical errors, constraint violations, and quality issues. |
| Plan Optimizer | Produces the corrected final schedule based on critique feedback. |

## Repository Structure

```text
src/study_planner/             Reusable Python package
  agents.py                    CrewAI agent factories
  config.py                    Output path and environment loading helpers
  difficulty.py                Sentence-transformer difficulty classifier
  evaluation.py                Structured plan parsing and scoring
  runner.py                    High-level variation runner and artifact saving
  tasks.py                     CrewAI task definitions
  tools.py                     Safe calculator and CrewAI tool wrappers
notebooks/study_planner_demo.ipynb
examples/sample_input.json     Example student profile
examples/sample_output.md      Example optimized schedule
tests/                         Unit and layout tests
pyproject.toml                 Project metadata and dependencies
requirements.txt               Editable install entrypoint
.env.example                   Example environment variables
```

Generated folders such as `outputs/`, `variation_outputs/`, and `tensorboard_logs/` are ignored because they are run artifacts, not source code.

## Requirements

- Python 3.10+
- OpenAI API key
- Serper API key
- Jupyter Notebook or Google Colab for the demo notebook

Install the project:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a local `.env` file from `.env.example`:

```bash
OPENAI_API_KEY=your-openai-api-key
SERPER_API_KEY=your-serper-api-key
STUDY_PLANNER_PROJECT_DIR=outputs
```

`load_api_keys()` reads `.env` first and only prompts for keys that are still missing. The real `.env` file is ignored by Git.

## How To Run

1. Install dependencies with `pip install -r requirements.txt`.
2. Create a `.env` file or be ready to enter missing API keys when prompted.
3. Open `notebooks/study_planner_demo.ipynb`.
4. Run the setup cells.
5. Run one temperature variation first to control API cost.
6. Review generated `.txt` and `.json` files under `outputs/variation_outputs/`.

For Google Colab, call `configure_colab_paths("your-drive-project-path")` instead of `configure_paths()` if you want artifacts saved to Drive.

## Examples

- Input profile: `examples/sample_input.json`
- Example output: `examples/sample_output.md`

The sample scenario consistently uses `Deep Learning` as the subject name across input, prompts, evaluation, and output.

## Testing

Run the test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

## What This Project Demonstrates

- Designing a practical multi-agent AI workflow with CrewAI.
- Combining LLM planning with deterministic helper tools.
- Using transformer embeddings for semantic classification.
- Evaluating LLM output quality across model parameter settings.
- Turning a notebook prototype into a testable Python project.

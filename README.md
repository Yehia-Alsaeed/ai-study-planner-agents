# AI Study Planner Agents

AI Study Planner Agents is a notebook-based multi-agent system that creates personalized study schedules from a student's subjects, exam dates, and available daily study hours.

The project combines CrewAI agents, GPT-4o mini, sentence-transformer embeddings, tool-augmented reasoning, and TensorBoard evaluation. It generates an initial study plan, critiques it against scheduling constraints, optimizes the result, and compares output quality across multiple model temperature settings.

## Key Features

- **Multi-agent planning workflow:** Uses specialized agents for student profiling, study plan generation, plan critique, and plan optimization.
- **Transformer-based difficulty classification:** Uses `all-MiniLM-L6-v2` sentence embeddings to classify subjects as hard, medium, or easy through semantic similarity.
- **Tool-augmented reasoning:** Adds external search, calculator support, and a custom subject difficulty classifier to improve planning decisions.
- **Temperature comparison:** Runs the same workflow with different GPT-4o mini temperature settings to compare stability, creativity, and schedule quality.
- **Evaluation pipeline:** Scores generated plans using completeness, constraint satisfaction, and quality metrics, then logs metrics to TensorBoard.

## System Workflow

1. The student provides subjects, exam dates, and daily available study hours.
2. The transformer embedding model classifies the difficulty of each subject.
3. The Student Profiler converts the raw request into structured planning data.
4. The Study Plan Generator creates a day-by-day schedule.
5. The Plan Critic checks the schedule for overloaded days, missing buffer days, incorrect exam handling, and weak subject prioritization.
6. The Plan Optimizer revises the schedule using the critic feedback.
7. The evaluation logic compares multiple run variations and logs results.

## Agent Architecture

| Agent | Responsibility |
| --- | --- |
| Student Profiler | Extracts subjects, exam dates, study capacity, and subject difficulty. |
| Study Plan Generator | Builds an initial day-by-day plan using planning constraints and tool support. |
| Plan Critic | Reviews the generated plan for logical errors, constraint violations, and quality issues. |
| Plan Optimizer | Produces the corrected final schedule based on critique feedback. |

## NLP Component

The notebook uses the `all-MiniLM-L6-v2` sentence-transformer model to embed subject names and compare them with reference examples for hard, medium, and easy subjects. This gives the planner a lightweight semantic difficulty signal without requiring a manually labeled training dataset.

## Example Scenario

The implemented notebook tests a student with four subjects: NLP, Deep Learning, Security, and Database. The exams are scheduled on different days, and the student has a fixed number of available study hours per day. The system then produces multiple schedule variations, critiques the plans, and compares the final outputs.

## Repository Structure

```text
study_planner.ipynb    Main notebook with agents, tools, variation runs, and evaluation logic
requirements.txt       Python dependencies used by the notebook
.env.example           Example environment variable names for required API keys
.gitignore             Excludes secrets, local environments, notebook checkpoints, and generated outputs
```

Generated folders such as `variation_outputs/` and `tensorboard_logs/` are intentionally ignored because they are run artifacts, not source code.

## Requirements

- Python 3.10+
- OpenAI API key
- Serper API key
- Jupyter Notebook or Google Colab

Install the Python packages:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a local `.env` file or enter the keys when prompted by the notebook:

```bash
OPENAI_API_KEY=your-openai-api-key
SERPER_API_KEY=your-serper-api-key
```

The real `.env` file is ignored by Git so credentials are not committed.

## How To Run

1. Open `study_planner.ipynb` in Google Colab or Jupyter.
2. Install the required packages from the setup cell or from `requirements.txt`.
3. Provide `OPENAI_API_KEY` and `SERPER_API_KEY` when prompted.
4. Run the notebook from top to bottom.
5. Review the generated study plan variations and TensorBoard metrics.

The original notebook was built for Google Colab and uses Google Drive paths for generated outputs. When running locally, update the output paths to local directories before executing the variation and evaluation cells.

## What This Project Demonstrates

- Designing a practical multi-agent AI workflow with CrewAI.
- Combining LLM planning with deterministic helper tools.
- Using transformer embeddings for semantic classification.
- Evaluating LLM output quality across model parameter settings.
- Building an end-to-end notebook pipeline from input parsing to final plan evaluation.

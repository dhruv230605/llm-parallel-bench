# LLM-Domain-Benchmark

This repository implements a research-style benchmark for **LLM-powered code generation in the domain of Parallel Programming in C++**. It includes a curated 10-task benchmark dataset, zero-shot generation scripts, an extended CodeBLUE-style evaluator, a modular multi-agent ReAct pipeline for iterative code improvement, and a standalone GitHub crawler used to mine and synthesize grounded benchmark tasks.

## Domain

The target domain is **parallel programming in C++**, covering shared-memory, lock-free, OpenMP, MPI, and concurrent data-structure tasks. The dataset currently includes:

1. Thread pool
2. OpenMP matrix multiplication
3. Producer-consumer queue
4. Lock-free queue
5. Parallel quicksort
6. Parallel prefix sum
7. MPI sum reduce
8. Parallel BFS
9. Work-stealing scheduler
10. Concurrent hash map

Each task lives under `dataset/task*/` and contains:

- `prompt.txt`: natural-language coding task
- `reference.cpp`: reference implementation
- `tests/test1.cpp`: executable validation harness
- `tests/run_tests.sh`: convenience wrapper around the Python evaluator

## Repository Layout

- `dataset/`: benchmark tasks and references
- `prompts/all_prompts.json`: central task catalog and compile metadata
- `generated_code/`: zero-shot and agent-generated solutions
- `evaluation/`: Pass@1, Extended CodeBLUE, and compile/run helpers
- `agent/`: planner, coder, tester, debugger, reviewer, and ReAct loop
- `scripts/`: zero-shot generation, agent generation, and full benchmark runners
- `results/`: CSV outputs for zero-shot, agent, and comparison tables
- `logs/`: benchmark, debugging, and experiment logs
- `model_registry.json`: provider/model mapping for OpenAI, Google, and NVIDIA NIM backends
- `web_crawler/`: standalone GitHub mining pipeline for discovering and synthesizing new benchmark tasks

## Web Crawler

The repository also includes a standalone `web_crawler/` module used to construct grounded benchmark tasks from real GitHub repositories. Its role is to support dataset creation by searching for parallel-programming artifacts such as issues, pull requests, and commits, ranking them by domain relevance and difficulty, and converting the strongest candidates into prompt-and-test benchmark tasks.

At a high level, the crawler does the following:

1. Loads search queries and domain/difficulty keywords from `web_crawler/config/`.
2. Searches GitHub for relevant C++ parallel-programming repositories.
3. Scrapes issues, pull requests, and commits from shortlisted repositories.
4. Parses those artifacts into candidate tasks.
5. Scores and filters candidates using keyword weights, complexity heuristics, and easy-task rejection.
6. Uses an LLM-backed normalization step to synthesize grounded benchmark tasks from shortlisted artifacts.
7. Writes the resulting dataset to `web_crawler/dataset/` along with cached provenance under `web_crawler/cache/`.

Important crawler outputs:

- `web_crawler/dataset/`: generated task folders with prompts, references, and tests
- `web_crawler/cache/repositories.json`: cached repository shortlist
- `web_crawler/cache/repos/`: cached repository artifact dumps
- `web_crawler/cache/details/`: deeper grounding metadata for selected artifacts
- `web_crawler/logs/crawler.log`: crawler execution log

In the current repository snapshot, `web_crawler/dataset/` has been synchronized to the same 10-task benchmark layout used by the main `dataset/` directory (`task1` to `task10`).

The crawler entry point is `web_crawler/main.py`. A typical run performs repository search, artifact scraping, candidate ranking, and dataset export in one pass.

To run it:

```bash
cd web_crawler
python main.py
```

Environment variables used by the crawler:

- `GITHUB_TOKEN`: recommended for higher GitHub API rate limits
- `OPENAI_API_KEY`: used for grounded task synthesis and normalization

This crawler is separate from the benchmark runner under `scripts/`, but it complements the main project by providing a reproducible way to expand or regenerate the benchmark from publicly available software engineering artifacts.

## Benchmark Methodology

### Zero-shot generation

`scripts/generate_zero_shot.py` loads prompts from `prompts/all_prompts.json`, queries one or more LLMs, and stores outputs under:

- `generated_code/<model>/<task>_attempt01.cpp`
- `generated_code/<model>/<task>.cpp`

By default, the script is configured for an OpenAI-compatible chat completion API via:

- `LLM_API_BASE`
- `LLM_API_KEY`

To support local smoke tests without external API access, the repository also supports:

- `--backend reference`

That mode copies each task's `reference.cpp` into the generated output directory so the full pipeline can be tested end to end.

### Pass@1

`evaluation/pass_at_1.py` implements the requested definition:

`Pass@1 = number of correct solutions / number of attempts`

For each task and model:

1. Up to 10 generated attempts are discovered
2. Each attempt is compiled
3. Each attempt is run against the task tests
4. Passing runs are counted
5. `Pass@1` is computed as `passes / attempts`

Compilation and execution are handled by `evaluation/compile_and_run.py`, which:

- Copies a candidate solution into a temporary build directory as `solution.cpp`
- Compiles the task test harness against that solution
- Adds `-fopenmp` for OpenMP tasks
- Uses `mpic++` and `mpiexec` for MPI tasks

### Extended CodeBLUE

`evaluation/codeblue.py` implements a practical, reproducible **extended CodeBLUE-style metric** for C++ parallel programming. It combines:

- N-gram overlap
- AST-structure proxy overlap
- Data-flow proxy overlap
- Syntax-profile overlap
- Domain keyword overlap

The domain-specific term weights are stored in `evaluation/domain_keywords.json` and include keywords such as:

- `thread`
- `mutex`
- `atomic`
- `parallel`
- `openmp`
- `mpi`
- `barrier`
- `condition_variable`
- `future`
- `async`
- `lock-free`
- `thread pool`
- `work stealing`

The final score is a weighted combination of:

- `ngram_score`
- `ast_score`
- `dataflow_score`
- `syntax_score`
- `keyword_score`

This implementation is intentionally lightweight and dependency-free, so it is easy to reproduce in teaching and project environments.

## Multi-Agent Architecture

The repository includes a ReAct-style coding pipeline under `agent/`:

1. `planner_agent.py`
   Breaks the task into implementation steps
2. `coder_agent.py`
   Produces the initial C++ solution
3. `tester_agent.py`
   Compiles and runs the code against task tests
4. `debugger_agent.py`
   Repairs failing code using test diagnostics
5. `reviewer_agent.py`
   Improves performance and parallel correctness
6. `react_loop.py`
   Runs the plan -> code -> test -> debug -> review loop
7. `agent_pipeline.py`
   Applies the loop across the dataset

Agent-generated outputs are stored under:

- `generated_code/agent/<model>/<task>.cpp`

## Models Used

The benchmark is model-agnostic, but the current repository is configured with the following registered models in `model_registry.json`:

- `gpt-4.1-mini`
- `gpt-4.1`
- `gpt-5.4-2026-03-05`
- `meta/llama-3.1-8b-instruct`
- `qwen/qwen2.5-coder-32b-instruct`
- `gemini-2.0-flash`

The committed benchmark results in this repository focus on these three models:

- `gpt-4.1-mini`
- `meta/llama-3.1-8b-instruct`
- `qwen/qwen2.5-coder-32b-instruct`

Provider credentials are resolved from environment variables declared in `model_registry.json`.

## Running the Benchmark

### 1. Smoke test with reference solutions

```bash
python scripts/run_benchmark.py --backend reference
```

This validates the repository structure, test harnesses, metric pipeline, and CSV generation without calling an external API.

### 2. Zero-shot generation only

```bash
python scripts/generate_zero_shot.py --models gpt-4.1-mini meta/llama-3.1-8b-instruct qwen/qwen2.5-coder-32b-instruct --attempts 10
python evaluation/evaluate_all.py --models gpt-4.1-mini meta/llama-3.1-8b-instruct qwen/qwen2.5-coder-32b-instruct --method Zero-shot --generated-root generated_code --output results/zero_shot_scores.csv
```

### 3. Agent generation only

```bash
python scripts/run_agent_generation.py --models gpt-4.1-mini meta/llama-3.1-8b-instruct qwen/qwen2.5-coder-32b-instruct --max-rounds 3
python evaluation/evaluate_all.py --models gpt-4.1-mini meta/llama-3.1-8b-instruct qwen/qwen2.5-coder-32b-instruct --method Agent --generated-root generated_code/agent --output results/agent_scores.csv
```

### 4. Full end-to-end benchmark

```bash
python scripts/run_benchmark.py --backend openai_compatible --zero-shot-models gpt-4.1-mini meta/llama-3.1-8b-instruct qwen/qwen2.5-coder-32b-instruct --agent-models gpt-4.1-mini meta/llama-3.1-8b-instruct qwen/qwen2.5-coder-32b-instruct --attempts 10 --max-rounds 3
```

## Results Format

The final CSV schema is:

- `Task Name`
- `Model Name`
- `Method`
- `Pass@1`
- `CodeBLUE Score`

Files:

- `results/zero_shot_scores.csv`
- `results/agent_scores.csv`

In the current committed results, the `Pass@1` column is used to store empirical success rate over 10 trials for the zero-shot and agent result tables shown below.

## Experimental Goal

The intended hypothesis for this benchmark is:

**A multi-agent ReAct-based coding pipeline produces stronger parallel C++ solutions than zero-shot generation, as measured by Pass@1 and Extended CodeBLUE.**

The repository provides the full scaffold required to test that hypothesis in a reproducible way.

## Benchmark Results

The committed result CSVs correspond to the current 10-task benchmark snapshot and summarize zero-shot versus multi-agent ReAct performance for `gpt-4.1-mini`, `meta/llama-3.1-8b-instruct`, and `qwen/qwen2.5-coder-32b-instruct`.

### Aggregate Comparison

```csv
Model,Metric,Zero-shot,Agent (3 rounds)
gpt-4.1-mini,Pass@1,0.40,0.97
gpt-4.1-mini,CodeBLEU,0.695,0.746
meta/llama-3.1-8b-instruct,Pass@1,0.22,0.65
meta/llama-3.1-8b-instruct,CodeBLEU,0.603,0.648
qwen/qwen2.5-coder-32b-instruct,Pass@1,0.39,0.93
qwen/qwen2.5-coder-32b-instruct,CodeBLEU,0.664,0.728
Overall,Pass@1,0.34,0.85
Overall,CodeBLEU,0.654,0.707
```

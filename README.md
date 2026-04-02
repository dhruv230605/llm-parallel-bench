# LLM-Domain-Benchmark

This repository implements a research-style benchmark for **LLM-powered code generation in the domain of Parallel Programming in C++**. It includes a curated dataset of domain tasks, zero-shot generation scripts, an extended CodeBLUE-style evaluator, and a modular multi-agent ReAct pipeline for iterative code improvement.

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
- `report.md`: compact research report template for the benchmark

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

The repository is model-agnostic. Default placeholders are:

- `gemini`
- `model2`
- `model3`

You can replace them with any real model IDs supported by your OpenAI-compatible endpoint.

## Running the Benchmark

### 1. Smoke test with reference solutions

```bash
python scripts/run_benchmark.py --backend reference
```

This validates the repository structure, test harnesses, metric pipeline, and CSV generation without calling an external API.

### 2. Zero-shot generation only

```bash
python scripts/generate_zero_shot.py --models gemini model2 model3 --attempts 10
python evaluation/evaluate_all.py --models gemini model2 model3 --method Zero-shot --generated-root generated_code --output results/zero_shot_scores.csv
```

### 3. Agent generation only

```bash
python scripts/run_agent_generation.py --models gemini --max-rounds 3
python evaluation/evaluate_all.py --models gemini --method Agent --generated-root generated_code/agent --output results/agent_scores.csv
```

### 4. Full end-to-end benchmark

```bash
python scripts/run_benchmark.py --zero-shot-models gemini model2 model3 --agent-models gemini --attempts 10 --max-rounds 3
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
- `results/comparison.csv`

These CSVs are initialized with headers and will be populated after running the pipeline.

## Experimental Goal

The intended hypothesis for this benchmark is:

**A multi-agent ReAct-based coding pipeline produces stronger parallel C++ solutions than zero-shot generation, as measured by Pass@1 and Extended CodeBLUE.**

The repository provides the full scaffold required to test that hypothesis in a reproducible way.

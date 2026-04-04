# Parallel C++ Web Crawler

This standalone module mines GitHub for hard parallel-programming tasks in C++, caches repository and artifact metadata locally, filters candidates with keyword- and complexity-based scoring, and exports a grounded benchmark dataset.

## How The Crawler Works

1. Load search queries and domain keywords from `config/`.
2. Search GitHub repositories and keep only domain-matching repositories, caching their URLs and scores in `cache/repositories.json`.
3. For each selected repository, fetch up to 10 issues, 10 pull requests, and 10 commits, caching each repo artifact dump under `cache/repos/`.
4. Convert GitHub metadata into candidate tasks and score them using keyword weights, complexity bonuses, and easy-task rejection.
5. Keep roughly 5-7 candidates per repository, then use `gpt-4o-mini` only on shortlisted candidates for normalization and final grounded task synthesis.
6. Fetch deeper grounding details only for final selected artifacts and cache them under `cache/details/`.
7. Export `prompt.txt`, `code.cpp`, optional `code.h`, tests, and provenance metadata under `dataset/task01` onward.

## Why These Tasks Are Hard For Zero-Shot Prompting

- They center on synchronization, shared-state coordination, and non-trivial parallel decomposition.
- Metadata scoring prioritizes race conditions, deadlocks, lock-free patterns, work stealing, OpenMP, MPI, and distributed coordination.
- Easy maintenance-only tasks such as docs, lint, typo, and formatting changes are rejected before LLM usage.
- Pull requests and commits with stronger grounding are preferred because they provide implementation evidence and patch context.

## Cache Layout

```text
cache/
  repositories.json
  repos/<repo>.json
  details/<artifact>.json
```

This prevents reusing the GitHub API for the same repositories and shortlisted artifacts on every run.

## Running The Crawler

```bash
python main.py
```

Environment variables:

- `GITHUB_TOKEN`: recommended for higher GitHub API limits.
- `OPENAI_API_KEY`: required for grounded task synthesis with `gpt-4o-mini`.

## Dataset Structure

```text
dataset/
  task01/
    prompt.txt
    code.cpp
    tests/test1.cpp
    tests/run_tests.sh
  ...
```

Repositories selected in the latest run: `20`.
Ranked candidates kept after filtering: `48`.
Tasks written in the latest run: `25`.

## Selected Repositories

- `uvhw/Bitcoin-Foundation` - https://github.com/uvhw/Bitcoin-Foundation - score 52.0
- `max0x7ba/atomic_queue` - https://github.com/max0x7ba/atomic_queue - score 50.0
- `mellcww/multitask-scheduler` - https://github.com/mellcww/multitask-scheduler - score 43.0
- `aguiarpedrof/cpp-producer-consumer-simulation` - https://github.com/aguiarpedrof/cpp-producer-consumer-simulation - score 41.0
- `bshoshany/thread-pool` - https://github.com/bshoshany/thread-pool - score 40.0
- `GeodeZister/ParallelQuickSort` - https://github.com/GeodeZister/ParallelQuickSort - score 38.0
- `lixiasky-back/tiny_coro-build_your_own_MN_scheduler` - https://github.com/lixiasky-back/tiny_coro-build_your_own_MN_scheduler - score 36.0
- `Aviisongit/Parallel-2D-Convolution-Engine-for-Deep-Learning-` - https://github.com/Aviisongit/Parallel-2D-Convolution-Engine-for-Deep-Learning- - score 36.0
- `spectre900/Parallel-Strassen-Algorithm` - https://github.com/spectre900/Parallel-Strassen-Algorithm - score 35.0
- `agarwal-ayushi/Parallel-Matrix-Vector-Multiplication` - https://github.com/agarwal-ayushi/Parallel-Matrix-Vector-Multiplication - score 35.0
- `shaanknight/Parallelised-CRAFTML` - https://github.com/shaanknight/Parallelised-CRAFTML - score 35.0
- `ramachav/Map-Reduce-Multi-Threaded` - https://github.com/ramachav/Map-Reduce-Multi-Threaded - score 35.0
- `mtrebi/thread-pool` - https://github.com/mtrebi/thread-pool - score 33.0
- `anant0301/openmp-bfs` - https://github.com/anant0301/openmp-bfs - score 33.0
- `inkooboo/thread-pool-cpp` - https://github.com/inkooboo/thread-pool-cpp - score 32.0
- `ptsouchlos/thread-pool` - https://github.com/ptsouchlos/thread-pool - score 32.0
- `giuseppegrieco/parallel-bfs` - https://github.com/giuseppegrieco/parallel-bfs - score 32.0
- `MatteoCeregini/parallel-bfs` - https://github.com/MatteoCeregini/parallel-bfs - score 32.0
- `jl2922/omp_hash_map` - https://github.com/jl2922/omp_hash_map - score 31.0
- `kaletap/bfs-cuda-gpu` - https://github.com/kaletap/bfs-cuda-gpu - score 30.0

## Notes

- This version intentionally does not use any seed task library.
- If live GitHub data or OpenAI synthesis is unavailable, the crawler may export fewer than 25 grounded tasks rather than silently fabricating fallback tasks.

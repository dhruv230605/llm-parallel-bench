#!/usr/bin/env bash
set -euo pipefail
SOLUTION=${1:-../reference.cpp}
python ../../../evaluation/compile_and_run.py --task task10 --candidate "$SOLUTION"

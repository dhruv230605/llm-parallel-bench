#!/usr/bin/env bash
set -euo pipefail
SOLUTION=${1:-../reference.cpp}
python ../../../evaluation/compile_and_run.py --task task7 --candidate "$SOLUTION"

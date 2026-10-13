#!/usr/bin/env bash
# Reproduce the n-body results (Table 2): BMS without damping on DW-4, LJ-13, and
# LJ-55, three seeds each. Run from the repository root after
# scripts/download_nbody_data.sh. Extra arguments are forwarded to Hydra, e.g.
#   bash scripts/nbody.sh damping=10
set -euo pipefail

for EXPERIMENT in dw4 lj13 lj55; do
  for SEED in 0 1 2; do
    python -u -m bms.experiment.train \
      experiment="$EXPERIMENT" \
      seed="$SEED" \
      name="${EXPERIMENT}_seed${SEED}" \
      root=outputs \
      "$@"
  done
done

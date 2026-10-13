#!/usr/bin/env bash
# Download the reference (long-run MCMC) samples for the n-body systems, as used
# by DEM / adjoint_samplers, into data/nbody/.
set -euo pipefail

DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/nbody"
mkdir -p "$DEST_DIR"

DEM_GITHUB="https://raw.githubusercontent.com/jarridrb/DEM/main/data"
for NPY in test_split_DW4.npy test_split_LJ13-1000.npy test_split_LJ55-1000-part1.npy; do
  curl -fL -o "$DEST_DIR/$NPY" "$DEM_GITHUB/$NPY"
done

echo "Files downloaded to $DEST_DIR"

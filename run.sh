#!/bin/bash
# run.sh
run_python_scripts() {
  local sigma="$1"
  local z="$2"

  python3 ws/ws_C10Des.py --randStd "$sigma" --ABgroup "$z" --gpuDevice 1
  python3 ws/ws_HAM10000Des.py   --randStd "$sigma" --ABgroup "$z" --gpuDevice 0
}
sigmas=(0.0 5.82e-11 1e-05 6e-4 1e-4 5e-3 1e-03 0.01 0.05 0.025 0.1 0.2 1.0)

for sigma in "${sigmas[@]}"; do
  for z in -1 1; do
    run_python_scripts "$sigma" "$z"
  done
done
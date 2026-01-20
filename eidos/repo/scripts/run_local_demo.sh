#!/bin/bash
set -e

# Run Local Demo
echo "Starting Eidos Local Demo..."
python -m eidos_brain.demo.demo_runner --mode local --config configs/demo_local.yaml

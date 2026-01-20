#!/bin/bash
set -e

# Run Stream Demo
echo "Starting Eidos Stream Demo..."
python -m eidos_brain.demo.demo_runner --mode stream --config configs/demo_stream.yaml

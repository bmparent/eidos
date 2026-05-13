#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
echo "Serving Eidos Life v0.2 at http://localhost:5173/experiments/eidos-life/"
python -m http.server 5173

#!/bin/bash
set -e
# Build and Publish
pip install build twine
python -m build
python -m twine upload dist/*

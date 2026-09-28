"""Regression: legacy proof imports must not hide root benchmark tests."""
from pathlib import Path
import subprocess
import sys


def test_legacy_package_does_not_hide_memory_benchmark():
    root = Path(__file__).resolve().parents[1]
    script = '''
import importlib.util, pathlib, runpy, sys
root=pathlib.Path.cwd()
for directory in (root/'eidos/proof',root/'proof'):
    spec=importlib.util.spec_from_file_location('proof',directory/'__init__.py',
        submodule_search_locations=[str(directory)])
    module=importlib.util.module_from_spec(spec)
    sys.modules['proof']=module
    spec.loader.exec_module(module)
    runpy.run_path(str(root/'tests/conftest.py'))
    assert pathlib.Path(importlib.util.find_spec('proof.memory_core').origin)==root/'proof/memory_core.py'
    assert pathlib.Path(importlib.util.find_spec('proof.sentinel_calibration_v1').origin)==root/'eidos/proof/sentinel_calibration_v1.py'
'''
    subprocess.run([sys.executable, '-c', script], cwd=root, check=True)

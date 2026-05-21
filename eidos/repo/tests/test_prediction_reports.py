from pathlib import Path
from eidos_brain.prediction.reports import write_report

def test_report_generation(tmp_path:Path):
    p=tmp_path/'r.md'; write_report(p,'T',['a']); assert 'T' in p.read_text()

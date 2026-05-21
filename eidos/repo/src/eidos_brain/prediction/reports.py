from __future__ import annotations
from pathlib import Path

def write_report(path:Path,title:str,lines:list[str])->None:
    path.write_text('\n'.join([f'# {title}','']+lines),encoding='utf-8')

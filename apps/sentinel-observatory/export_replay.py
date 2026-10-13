"""Export a deliberately offline, clearly labeled replay of collected evidence.

This export never starts a collector and never presents the capture as live.
The local service already allowlists evidence fields and omits bodies, cookies,
authorization headers and client IP addresses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import urllib.request


def build_replay(snapshot: dict, output: Path) -> None:
    if not isinstance(snapshot.get('events'), list) or not snapshot['events']:
        raise ValueError('No real recorded events to export')
    snapshot = dict(snapshot)
    snapshot['mode'] = 'recorded'
    snapshot['export_notice'] = 'Recorded real HTTP measurements. Offline replay; no live connection or current protection claim.'
    web = Path(__file__).resolve().parent / 'web'
    html = (web / 'index.html').read_text(encoding='utf-8')
    css = (web / 'styles.css').read_text(encoding='utf-8')
    js = (web / 'app.js').read_text(encoding='utf-8')
    payload = json.dumps(snapshot, ensure_ascii=True, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    html, css_count = re.subn(r'<link\b[^>]*href=["\'][^"\']*styles\.css["\'][^>]*>', lambda _: '<style>' + css + '</style>', html)
    html, js_count = re.subn(r'<script\b[^>]*src=["\'][^"\']*app\.js["\'][^>]*>\s*</script>', lambda _: '<script>window.__OBSERVATORY_RECORDING__=' + payload + ';</script><script>' + js.replace('</script', '<\\/script') + '</script>', html)
    if css_count != 1 or js_count != 1:
        raise ValueError('Unexpected web entrypoint; refuse a partially connected export')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, help='Previously exported snapshot JSON; otherwise read localhost:8765')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.snapshot:
        snapshot = json.loads(args.snapshot.read_text(encoding='utf-8'))
    else:
        with urllib.request.urlopen('http://127.0.0.1:8765/api/snapshot', timeout=10) as response:
            snapshot = json.load(response)
    build_replay(snapshot, args.output)
    print(json.dumps({'output': str(args.output.resolve()), 'events': len(snapshot['events']), 'mode': 'recorded'}))


if __name__ == '__main__':
    main()

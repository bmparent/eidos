"""Finite, real-data acceptance run. Makes the app's fixed read-only HEAD probes.

No fixtures enter the journal. Output contains measured public-response metadata
only. This deliberately short experiment does not qualify a security product.
"""
from __future__ import annotations
import argparse
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path
from http.server import ThreadingHTTPServer
from server import make_handler
from telemetry import Observatory, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=int, default=420)
    parser.add_argument('--interval', type=float, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    observatory = Observatory(str(args.output / 'observations.sqlite3'), args.interval)
    http = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(observatory))
    http.daemon_threads = True
    base = f'http://127.0.0.1:{http.server_port}'
    server_thread = threading.Thread(target=http.serve_forever, daemon=True)
    server_thread.start()
    observatory.start()
    timings = []
    model_timings = []
    started = time.monotonic()
    probe_response = None
    last_report = -30
    try:
        while time.monotonic()-started < args.duration:
            t = time.perf_counter()
            with urllib.request.urlopen(base+'/api/snapshot', timeout=8) as response:
                snapshot = json.load(response)
            elapsed_ms = (time.perf_counter()-t)*1000
            timings.append(elapsed_ms)
            if snapshot['model'].get('state') in ('ready','loading') or snapshot['coverage']['analysis_queue']:
                model_timings.append(elapsed_ms)
            if probe_response is None and snapshot['events']:
                request = urllib.request.Request(base+'/api/probe', data=b'{}', method='POST', headers={'Origin':base, 'Content-Type':'application/json'})
                with urllib.request.urlopen(request, timeout=8) as response:
                    probe_response = {'status':response.status,'body':json.load(response),'requested_at':utc_now()}
            # Persist the actual current response so other review processes can
            # inspect evidence without needing this process's network namespace.
            temporary = args.output/'snapshot.tmp'
            temporary.write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
            temporary.replace(args.output/'snapshot.json')
            elapsed = time.monotonic()-started
            if elapsed-last_report >= 30:
                print(json.dumps({'elapsed_s':round(elapsed),'observations':len(snapshot['events']),
                    'model':snapshot['model']['state'],'model_results':sum(e['analysis']['kind']=='model' for e in snapshot['events']),
                    'probe':[(e['status'],e['analysis']['kind']) for e in snapshot['events'] if e['kind']=='controlled_probe'],
                    'api_latest_ms':round(elapsed_ms,1)}),flush=True)
                last_report=elapsed
            time.sleep(1)
    finally:
        observatory.stop()
        http.shutdown()
        http.server_close()
    snapshot=observatory.snapshot()
    snapshot['collection_enabled']=False
    snapshot['capture_complete']=True
    snapshot['mode']='recorded'
    (args.output/'snapshot.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
    def p95(values):
        return sorted(values)[min(len(values)-1,int(.95*len(values)))] if values else None
    events=snapshot['events']
    receipt={'started_at':observatory.started_at,'completed_at':utc_now(),'mode':'finite_real_live_connection_test',
        'duration_seconds':round(time.monotonic()-started,2),'probe_request':probe_response,
        'observations':len(events),'status_counts':{str(k):sum(e['status']==k for e in events) for k in set(e['status'] for e in events)},
        'model_results':sum(e['analysis']['kind']=='model' for e in events),
        'api_requests':len(timings),'api_p50_ms':statistics.median(timings),'api_p95_ms':p95(timings),
        'api_max_ms':max(timings),'api_p95_during_analysis_ms':p95(model_timings),
        'journal':observatory.store.verify(),'brain':snapshot['brain'],
        'limitations':['One outbound proxy vantage; not origin performance','Short observational window; no incident labels',
            'No browser rendering claimed by this API run','Model catalogue ranking does not establish causal or security reasoning']}
    (args.output/'live-acceptance.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()

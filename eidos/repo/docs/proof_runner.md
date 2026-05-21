# Proof Runner

Install in editable mode from `eidos/repo`:

```bash
python -m pip install -e .
```

Run hardened smoke suite:

```bash
python -m eidos_brain.proof.run_proof --suite smoke --seeds 0,1 --frames 10000 --families s1_backdoor,s6_noise_thrash --out ../artifacts/eidos_hidden_structure_proof_smoke_hardened
```

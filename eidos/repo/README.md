# Eidos Brain

## Quickstart (Local)

1. Create a virtual environment:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```
2. Install the package:
   ```bash
   pip install -e .
   ```
3. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
4. Run the demo:
   ```bash
   ./scripts/run_local_demo.sh
   ```

## Quickstart (Docker)

```bash
docker-compose up --build
```

## Cloud Run

```bash
./scripts/deploy_cloud_run.sh
```

#!/bin/bash
set -e

PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
REGION="us-central1"
SERVICE_NAME="eidos-brain"
IMAGE_TAG="gcr.io/$PROJECT_ID/$SERVICE_NAME:latest"

echo "Deploying to Cloud Run ($PROJECT_ID)..."

# Build and Push
gcloud builds submit --tag $IMAGE_TAG .

# Deploy API
gcloud run deploy $SERVICE_NAME-api \
    --image $IMAGE_TAG \
    --platform managed \
    --region $REGION \
    --command "eidos-api" \
    --allow-unauthenticated \
    --set-env-vars HIVE_BACKEND=GCS

# Deploy Sentinel (Worker)
gcloud run deploy $SERVICE_NAME-sentinel \
    --image $IMAGE_TAG \
    --platform managed \
    --region $REGION \
    --command "eidos" \
    --args "--loop" \
    --no-allow-unauthenticated \
    --set-env-vars HIVE_BACKEND=GCS

echo "Deployment complete."

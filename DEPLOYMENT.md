# Azure Container App Deployment Guide

This guide explains how to deploy sklearn-playground to Azure Container Apps.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Resource Group: rg-skplayground-dev                        │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Container Apps Environment                           │   │
│  │  └── Container App (sklearn-playground)              │   │
│  │       • Port: 8501                                   │   │
│  │       • Auto-scaling: 1-3 replicas                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  │
│  │ Azure Container Registry │  │ PostgreSQL Flexible     │  │
│  │  └── sklearn-playground  │  │  └── Database:          │  │
│  │      :latest             │  │      skplayground       │  │
│  └─────────────────────────┘  └─────────────────────────┘  │
│                                                             │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  │
│  │ Storage Account         │  │ Key Vault               │  │
│  │  └── Containers:        │  │  └── Secrets:           │  │
│  │      models, datasets   │  │      postgres-password  │  │
│  └─────────────────────────┘  └─────────────────────────┘  │
│                                                             │
│  ┌─────────────────────────┐                                │
│  │ Log Analytics Workspace │                                │
│  └─────────────────────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Azure Subscription** with Contributor access
2. **Azure CLI** installed and logged in (`az login`)
3. **Docker** installed and running
4. **jq** installed (for JSON parsing in scripts)

## Deployment Options

### Option 1: GitHub Actions (Recommended)

Automated CI/CD pipeline that deploys on every push to `main`.

#### Setup Steps

1. **Create an Azure Service Principal**

   ```bash
   # Get your subscription ID
   SUBSCRIPTION_ID=$(az account show --query id -o tsv)
   echo "Subscription ID: $SUBSCRIPTION_ID"

   # Create service principal with Contributor role
   az ad sp create-for-rbac \
     --name "github-skplayground" \
     --role Contributor \
     --scopes /subscriptions/$SUBSCRIPTION_ID
   ```

   This outputs JSON like:
   ```json
   {
     "appId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
     "displayName": "github-skplayground",
     "password": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
     "tenant": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
   }
   ```

2. **Configure GitHub Secrets**

   Go to your repository → Settings → Secrets and variables → Actions → New repository secret

   Add these 4 secrets:

   | Secret Name | Value from step 1 |
   |-------------|-------------------|
   | `AZURE_CLIENT_ID` | `appId` value |
   | `AZURE_CLIENT_SECRET` | `password` value |
   | `AZURE_TENANT_ID` | `tenant` value |
   | `AZURE_SUBSCRIPTION_ID` | Your subscription ID |

3. **Trigger Deployment**

   - Push to `main` branch, or
   - Go to Actions → "Deploy to Azure Container Apps" → Run workflow

4. **Access Your App**

   After deployment, the workflow will display the application URL in the summary.

### Option 2: Manual Deployment

Deploy directly from your local machine.

```bash
# Navigate to project root
cd /workspaces/sklearnplay

# Make script executable
chmod +x infra/deploy.sh

# Run deployment (defaults to 'dev' environment)
./infra/deploy.sh

# Or specify environment
./infra/deploy.sh dev

# Or specify custom resource group
./infra/deploy.sh dev rg-my-custom-name
```

The script will:
1. Create the resource group
2. Generate a secure PostgreSQL password
3. Deploy all Azure resources via Bicep
4. Build and push the Docker image
5. Update the Container App

## Environment Variables

The following environment variables are automatically configured in Azure:

| Variable | Description |
|----------|-------------|
| `ENVIRONMENT` | Deployment environment (dev/staging/prod) |
| `POSTGRES_HOST` | PostgreSQL server hostname |
| `POSTGRES_PORT` | PostgreSQL port (5432) |
| `POSTGRES_DB` | Database name (skplayground) |
| `POSTGRES_USER` | Database username |
| `POSTGRES_PASSWORD` | Database password (from secret) |
| `POSTGRES_SSL_MODE` | SSL mode (require) |
| `AZURE_STORAGE_CONNECTION_STRING` | Storage account connection |
| `AZURE_STORAGE_ACCOUNT` | Storage account name |
| `ENABLE_AUTH` | Authentication enabled (false) |
| `ENABLE_EXPERIMENTS` | Experiment tracking enabled (true) |
| `ENABLE_MODEL_STORAGE` | Model storage enabled (true) |

## Estimated Costs

| Resource | SKU | Monthly Cost |
|----------|-----|--------------|
| Container App | 0.5 vCPU, 1Gi (westus2) | ~$10-15 |
| PostgreSQL | Standard_B1ms | ~$15 |
| Storage Account | Standard_LRS | ~$1-5 |
| Container Registry | Basic | ~$5 |
| Log Analytics | PerGB2018 | ~$2-5 |
| **Total** | | **~$35-45** |

## Troubleshooting

### View Container App Logs

```bash
# Stream logs
az containerapp logs show \
  --name ca-skplayground-dev \
  --resource-group rg-skplayground-dev \
  --follow

# View recent logs
az containerapp logs show \
  --name ca-skplayground-dev \
  --resource-group rg-skplayground-dev \
  --tail 100
```

### Check Container App Status

```bash
az containerapp show \
  --name ca-skplayground-dev \
  --resource-group rg-skplayground-dev \
  --query "properties.runningStatus"
```

### Restart Container App

```bash
az containerapp revision restart \
  --name ca-skplayground-dev \
  --resource-group rg-skplayground-dev \
  --revision $(az containerapp revision list \
    --name ca-skplayground-dev \
    --resource-group rg-skplayground-dev \
    --query "[0].name" -o tsv)
```

### Connect to PostgreSQL

```bash
# Get the password from Key Vault
az keyvault secret show \
  --vault-name kv-skplayground-dev \
  --name postgres-password \
  --query value -o tsv

# Connect using psql
psql "host=psql-skplayground-dev.postgres.database.azure.com \
      port=5432 \
      dbname=skplayground \
      user=pgadmin \
      sslmode=require"
```

## Cleanup

To delete all deployed resources:

```bash
# Delete the entire resource group
az group delete --name rg-skplayground-dev --yes --no-wait
```

## Local Development

For local development with Docker Compose:

```bash
# Start local environment
docker-compose up -d

# Access the app at http://localhost:8501

# Stop local environment
docker-compose down
```

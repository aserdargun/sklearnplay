#!/bin/bash
# Deployment script for sklearn-playground to Azure
# Usage: ./deploy.sh <environment> [resource-group]

set -euo pipefail

# Configuration
ENVIRONMENT="${1:-dev}"
RESOURCE_GROUP="${2:-rg-skplayground-${ENVIRONMENT}}"
LOCATION="${LOCATION:-eastus}"
BASE_NAME="${BASE_NAME:-skplayground}"
ACR_NAME="${ACR_NAME:-acrskplayground${ENVIRONMENT}}"

echo "========================================"
echo "sklearn-playground Azure Deployment"
echo "========================================"
echo "Environment:    ${ENVIRONMENT}"
echo "Resource Group: ${RESOURCE_GROUP}"
echo "Location:       ${LOCATION}"
echo "========================================"

# Check required tools
command -v az >/dev/null 2>&1 || { echo "Azure CLI required but not installed. Aborting." >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker required but not installed. Aborting." >&2; exit 1; }

# Login check
az account show >/dev/null 2>&1 || { echo "Please login with 'az login' first." >&2; exit 1; }

echo ""
echo "Step 1: Creating Resource Group..."
az group create \
    --name "${RESOURCE_GROUP}" \
    --location "${LOCATION}" \
    --tags environment="${ENVIRONMENT}" application="sklearn-playground"

echo ""
echo "Step 2: Creating Azure Container Registry..."
az acr create \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${ACR_NAME}" \
    --sku Basic \
    --admin-enabled true

# Get ACR credentials
ACR_LOGIN_SERVER=$(az acr show --name "${ACR_NAME}" --query loginServer -o tsv)
ACR_USERNAME=$(az acr credential show --name "${ACR_NAME}" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "${ACR_NAME}" --query "passwords[0].value" -o tsv)

echo ""
echo "Step 3: Building and pushing Docker image..."
docker build -t "${ACR_LOGIN_SERVER}/sklearn-playground:latest" .
echo "${ACR_PASSWORD}" | docker login "${ACR_LOGIN_SERVER}" -u "${ACR_USERNAME}" --password-stdin
docker push "${ACR_LOGIN_SERVER}/sklearn-playground:latest"

echo ""
echo "Step 4: Generating secure passwords..."
POSTGRES_PASSWORD=$(openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c 24)

echo ""
echo "Step 5: Deploying infrastructure with Bicep..."

# Check if Azure Entra ID is configured
AZURE_TENANT_ID="${AZURE_TENANT_ID:-}"
AZURE_CLIENT_ID="${AZURE_CLIENT_ID:-}"
AZURE_CLIENT_SECRET="${AZURE_CLIENT_SECRET:-}"

az deployment group create \
    --resource-group "${RESOURCE_GROUP}" \
    --template-file "infra/main.bicep" \
    --parameters environment="${ENVIRONMENT}" \
    --parameters baseName="${BASE_NAME}" \
    --parameters postgresAdminPassword="${POSTGRES_PASSWORD}" \
    --parameters containerImage="${ACR_LOGIN_SERVER}/sklearn-playground:latest" \
    --parameters azureTenantId="${AZURE_TENANT_ID}" \
    --parameters azureClientId="${AZURE_CLIENT_ID}" \
    --parameters azureClientSecret="${AZURE_CLIENT_SECRET}"

echo ""
echo "Step 6: Getting deployment outputs..."
OUTPUTS=$(az deployment group show \
    --resource-group "${RESOURCE_GROUP}" \
    --name main \
    --query properties.outputs)

APP_URL=$(echo "${OUTPUTS}" | jq -r '.containerAppUrl.value')
POSTGRES_FQDN=$(echo "${OUTPUTS}" | jq -r '.postgresServerFqdn.value')
KEYVAULT_NAME=$(echo "${OUTPUTS}" | jq -r '.keyVaultName.value')

echo ""
echo "========================================"
echo "Deployment Complete!"
echo "========================================"
echo ""
echo "Application URL: ${APP_URL}"
echo "PostgreSQL FQDN: ${POSTGRES_FQDN}"
echo "Key Vault:       ${KEYVAULT_NAME}"
echo ""
echo "PostgreSQL Password has been stored in Key Vault."
echo ""
echo "Next Steps:"
echo "1. Configure Azure Entra ID app registration for authentication"
echo "2. Update redirect URI in Entra ID to: ${APP_URL}/auth/callback"
echo "3. Run database migrations if needed"
echo ""

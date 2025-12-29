#!/bin/bash
# Deployment script for sklearn-playground to Azure
# Usage: ./deploy.sh [environment] [resource-group]

set -euo pipefail

# Configuration
ENVIRONMENT="${1:-dev}"
RESOURCE_GROUP="${2:-rg-skplayground-${ENVIRONMENT}}"
LOCATION="${LOCATION:-westus2}"
BASE_NAME="${BASE_NAME:-skplayground}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "========================================"
echo "sklearn-playground Azure Deployment"
echo "========================================"
echo "Environment:    ${ENVIRONMENT}"
echo "Resource Group: ${RESOURCE_GROUP}"
echo "Location:       ${LOCATION}"
echo "Image Tag:      ${IMAGE_TAG}"
echo "========================================"

# Check required tools
command -v az >/dev/null 2>&1 || { echo "Azure CLI required but not installed. Aborting." >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker required but not installed. Aborting." >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "jq required but not installed. Aborting." >&2; exit 1; }

# Login check
az account show >/dev/null 2>&1 || { echo "Please login with 'az login' first." >&2; exit 1; }

echo ""
echo "Step 1: Creating Resource Group..."
az group create \
    --name "${RESOURCE_GROUP}" \
    --location "${LOCATION}" \
    --tags environment="${ENVIRONMENT}" application="sklearn-playground"

echo ""
echo "Step 2: Generating secure PostgreSQL password..."
POSTGRES_PASSWORD=$(openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c 24)

echo ""
echo "Step 3: Deploying infrastructure with Bicep (includes ACR)..."
az deployment group create \
    --resource-group "${RESOURCE_GROUP}" \
    --template-file "infra/main.bicep" \
    --parameters environment="${ENVIRONMENT}" \
    --parameters baseName="${BASE_NAME}" \
    --parameters postgresAdminPassword="${POSTGRES_PASSWORD}" \
    --parameters imageTag="${IMAGE_TAG}"

echo ""
echo "Step 4: Getting deployment outputs..."
OUTPUTS=$(az deployment group show \
    --resource-group "${RESOURCE_GROUP}" \
    --name main \
    --query properties.outputs)

ACR_NAME=$(echo "${OUTPUTS}" | jq -r '.containerRegistryName.value')
ACR_LOGIN_SERVER=$(echo "${OUTPUTS}" | jq -r '.containerRegistryLoginServer.value')
APP_URL=$(echo "${OUTPUTS}" | jq -r '.containerAppUrl.value')
POSTGRES_FQDN=$(echo "${OUTPUTS}" | jq -r '.postgresServerFqdn.value')
KEYVAULT_NAME=$(echo "${OUTPUTS}" | jq -r '.keyVaultName.value')

echo ""
echo "Step 5: Getting ACR credentials..."
ACR_USERNAME=$(az acr credential show --name "${ACR_NAME}" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "${ACR_NAME}" --query "passwords[0].value" -o tsv)

echo ""
echo "Step 6: Building and pushing Docker image..."
docker build -t "${ACR_LOGIN_SERVER}/sklearn-playground:${IMAGE_TAG}" .
echo "${ACR_PASSWORD}" | docker login "${ACR_LOGIN_SERVER}" -u "${ACR_USERNAME}" --password-stdin
docker push "${ACR_LOGIN_SERVER}/sklearn-playground:${IMAGE_TAG}"

echo ""
echo "Step 7: Updating Container App with new image..."
CONTAINER_APP_NAME=$(echo "${OUTPUTS}" | jq -r '.containerAppName.value')
az containerapp update \
    --name "${CONTAINER_APP_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --image "${ACR_LOGIN_SERVER}/sklearn-playground:${IMAGE_TAG}"

echo ""
echo "========================================"
echo "Deployment Complete!"
echo "========================================"
echo ""
echo "Application URL: ${APP_URL}"
echo "PostgreSQL FQDN: ${POSTGRES_FQDN}"
echo "Key Vault:       ${KEYVAULT_NAME}"
echo "Container Registry: ${ACR_LOGIN_SERVER}"
echo ""
echo "PostgreSQL Password has been stored in Key Vault."
echo ""
echo "The application should be available in a few minutes at:"
echo "  ${APP_URL}"
echo ""

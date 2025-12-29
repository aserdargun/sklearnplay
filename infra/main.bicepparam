using './main.bicep'

// Example parameters file for development environment
// Copy this file and customize for your environment

param environment = 'dev'
param baseName = 'skplayground'
param postgresAdminUser = 'pgadmin'
param postgresAdminPassword = '' // Set via --parameters at deployment time

// Azure Entra ID (optional - leave empty to disable auth)
param azureTenantId = ''
param azureClientId = ''
param azureClientSecret = '' // Set via --parameters at deployment time

// Container image - update after pushing to ACR
param containerImage = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'

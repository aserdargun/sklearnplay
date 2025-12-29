using './main.bicep'

// Parameters file for development environment
// Note: postgresAdminPassword should be passed at deployment time via CLI

param environment = 'dev'
param baseName = 'skplayground'
param postgresAdminUser = 'pgadmin'
param postgresAdminPassword = '' // Set via --parameters at deployment time
param imageTag = 'latest'

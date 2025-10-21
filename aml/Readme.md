# Azure ML Singularity Jobs

## Environment
```
pip install azure-ai-ml==1.5.0a20230215003 --extra-index-url https://pkgs.dev.azure.com/azure-sdk/public/_packaging/azure-sdk-for-python/pypi/simple/
```

```
az ml environment create -f <environment-yaml-path> -g <resource-group-name> -w <workspace-name>
```

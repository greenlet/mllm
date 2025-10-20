# Azure ML Singularity Jobs

## Environment
```
pip install azure-ai-ml==1.5.0a20230215003 --extra-index-url https://pkgs.dev.azure.com/azure-sdk/public/_packaging/azure-sdk-for-python/pypi/simple/
```

```
az ml environment create -f "q:\prog\mllm\aml\env\mllm_train\environment.yaml" -g Multimedia-Singularity-RG-01 -w multimedia-singularity-ws01-eus
```

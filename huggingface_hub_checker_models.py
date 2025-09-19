from huggingface_hub import HfApi

api = HfApi()
models_gen = api.list_models(filter="cross-encoder")

models_list = list(models_gen)

for m in models_list[:40]:
    print(m.modelId)
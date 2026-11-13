Original README here:
https://huggingface.co/google/embeddinggemma-2/raw/main/README.md

To download model:
```
$ pip install huggingface_hub
$ hf download google/embeddinggemma-2 --local-dir /local/dir/path
```

## Modification Notice

The original `model.safetensors` checkpoint from `google/embeddinggemma-2` was
sharded into `<= 100 MB` `.safetensors` files
(`shard-00001-of-00017.safetensors` through `shard-00017-of-00017.safetensors`,
with large tensors split across shards) and indexed by `shards.index.json`.

## License

EmbeddingGemma 2 (`google/embeddinggemma-2`) is licensed under the Apache
License, Version 2.0 (see the repository root `LICENSE` file and
https://ai.google.dev/gemma/apache_2). Deployments must adhere to the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

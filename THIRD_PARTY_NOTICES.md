# Third-party model code

`captcha_lab/hf_models.py` adapts the architecture and preprocessing published by
Graf-J in the following model repositories (declared MIT on the model cards):

- https://huggingface.co/Graf-J/captcha-crnn-base
- https://huggingface.co/Graf-J/captcha-crnn-finetuned
- https://huggingface.co/Graf-J/captcha-conv-transformer-base
- https://huggingface.co/Graf-J/captcha-conv-transformer-finetuned

Exact revisions and source-file SHA-256 hashes are recorded in
`research/hf_lock.json`. Weights are downloaded separately and are not redistributed.
Review the upstream license and training-data terms before redistribution.

No remote Python code is executed by these adapters. Only config, vocabulary and
safetensors weights are loaded. Architecture differences fail strict loading.

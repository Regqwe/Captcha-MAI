# Third-party assets

The runtime uses RapidOCR and ONNX Runtime to run the PP-OCRv6 small recognizer.
Weights are downloaded separately by RapidOCR and are not redistributed here.
The pinned runtime and weight checksum are in research/model.json.
Model card: https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_safetensors

The 1984 evaluation images originate from project-sloth/captcha-images,
revision eeaf2b6ec9086645f270f7e2aaa9ea90730683de; its dataset card declares WTFPL.
Source: https://huggingface.co/datasets/project-sloth/captcha-images
Nine target crops originate from the user's supplied screenshots.

The `upscale.py` script takes in an image dataset and upscales them and uploads it back to huggingface with a `_upscaled` suffix.

Upscaling takes about ~4GBs of vRAM.

## Upscale.py CLI Usage
```Python
# activate the env
# 1. CLI help
python3 upscale.py upscale --help

# 2. trigger upscale
python3 upscale.py upscale -d <dataset-name> -v <dataset-version> -h <huggingface-token>

# -d: dataset name (or prefix part of it. eg. if your dataset is dushj98/waikato_aerial_2017_synthetic_v1, -d is dushj98/waikato_aerial_2017_synthetic). Default value: dushj98/waikato_aerial_2017_synthetic
# -v: version of the dataset. (based on above eg, -v is _v1 (yes, with the underscore))
# -h: huggingface token for dushj98 
```

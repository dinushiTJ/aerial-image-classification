Generation of images locally using the `lora.py` script is way faster than generating images in Kaggle. Inference is much faster on RTX3080.

The generated images are pushed to 13 partial synthetic datasets and needs to be merged using the `merge_lora.py` script.

Both scripts provide an easy to use CLI.

## Lora.py CLI Usage
- For each generation attempt, you have to properly version them to create separate synthetic datasets or it will result in replacing an existing dataset on the huggingface hub if you accidentally specify an existing version tag. Version tags are just a prefix to identify the synthetic datasets, so you can version them like `_v10`, `_v11` or even like `_test` to version them. Versions already used and you must not override: `_v0`, `_v1`, `_v2`.

How to trigger a synthetic data generation for a new synthetic data version:
```Python
# activate your env first, and navigate to generation dir
# 1. for CLI usage assistance
python3 lora.py generate --help  

# 2. run a generation
# specify -n to prevent pushing partial datasets to huggingface
# specify -c to run on CPU if you do not have a GPU or you need to force the script to do so
python3 lora.py generate -v _v100 -h <huggingface-token> -i 50 -s 50  

# -v: version
# -h: huggingface token (only works for the account `dushj98` or other allowed users)
# -i: number of images to generate per prompt. there are 20 prompts for each class, i.e. 260 prompts
# -s: number of inference steps, the higher you go, the longer it takes. For these LoRA adaptors, ~25 - ~60 works the same.
```

The `similarity.py` script is designed to calculate distribution similarities of two image datasets using FID and CMMD scores.

## Similarity.py CLI Usage
```Python
# activate the env
# 1. CLI help
python3 similarity.py --help
python3 similarity.py prep --help
python3 similarity.py prep real --help
python3 similarity.py prep synthetic --help
python3 similarity.py fid --help
python3 similarity.py cmmd --help

# 2. Prepare the real dataset
# -f is optional, if present, it discards any already prepared real dataset in the local dir and re-prepares it
python3 similarity.py prep real -h <huggingface-token>
# -h: dushj98 huggingface token
# -f: if present, force-prepares the real dataset
# this will create a real and real_all dirs in the current working dir.

# 3. Preapare a synthetic dataset
python3 similarity.py prep synthetic -h <huggingface-token> -v <dataset-version>
# -f is optional, -v is optional too, but is provided under usual cases, -h is bound to dushj98
# this will create a synthetic{version} and synthetic{version}_all dirs in the current working dir
# it picks 8658 images randomly and each class in your synthetic dataset must have 1000 images per each class.
# if not, the preparation will fail.

# 4. Calculate scores
python3 similarity.py fid -v <dataset-version>
# by default, this will do nothing, you have to specify:
#   -c: to calculate class-wise scores
#   -d: to calculate dataset-wide score
#   -a: for both 

python3 similarity.py cmmd -v <dataset-version>
# by default, this will do nothing, you have to specify:
#   -c: to calculate class-wise scores
#   -d: to calculate dataset-wide score
#   -a: for both

# -v: version of the dataset. (f your dataset is dushj98/waikato_aerial_2017_synthetic_v1, -v is _v1 (yes, with the underscore))
# -h: huggingface token for dushj98 
```

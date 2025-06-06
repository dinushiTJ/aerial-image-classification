import secrets
import os

def generate_unique_seeds(count=4, exclude=None):
    if exclude is None:
        exclude = set()
    else:
        exclude = set(exclude)

    seeds = set()
    while len(seeds) < count:
        seed = secrets.randbelow(2**32)
        if seed not in exclude:
            seeds.add(seed)
    return list(seeds)

# Exclude seed 42 (already used)
random_seeds = generate_unique_seeds(count=4, exclude={42})

# Define output path
output_path = os.path.expanduser("~/research/code/waikato_aerial/trainclassif/random_seeds.txt")

# Write seeds to file
with open(output_path, "w") as f:
    for seed in random_seeds:
        f.write(f"{seed}\n")

print(f"Random seeds saved to: {output_path}")

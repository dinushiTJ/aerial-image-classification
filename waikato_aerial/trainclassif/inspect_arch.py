import torch
import torchvision.models as models
import json

def list_all_layers(model):
    count = 0
    layer_names = []
    # Iterate over all layers/modules in the model
    for name, layer in model.named_modules():
        # Check for Conv2d and Linear layers
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            count += 1
            layer_names.append(name)
    return count, layer_names

def get_model_layers(model_name):
    """Loads a model and returns its layers as a list."""
    try:
        if model_name == "mobilenet_v2":
            model = models.mobilenet_v2(weights=None)
        elif model_name == "resnet50":
            model = models.resnet50(weights=None)
        elif model_name == "efficientnet_b2":
            model = models.efficientnet_b2(weights=None)
        elif model_name == "efficientnetv2_m":
            model = models.efficientnet_v2_m(weights=None)
        elif model_name == "vit_b_16":
            model = models.vit_b_16(weights=None)
        else:
            print(f"Model '{model_name}' not recognized.")
            return None

        # Collect layer count and layer names
        num_layers, layer_names = list_all_layers(model)

        return {
            "num_layers": num_layers,
            "layers": layer_names
        }

    except Exception as e:
        print(f"An error occurred while loading {model_name}: {e}")
        return None

# List of models you want to inspect
models_to_inspect = ["mobilenet_v2", "resnet50", "efficientnet_b2", "efficientnetv2_m", "vit_b_16"]

# Final dictionary to dump
models_architecture = {}

for model_name in models_to_inspect:
    model_info = get_model_layers(model_name)
    if model_info is not None:
        models_architecture[model_name] = model_info

# Dump to JSON file
with open("model_arch.json", "w") as f:
    json.dump(models_architecture, f, indent=4)

print("Model architectures dumped to 'model_arch.json'.")

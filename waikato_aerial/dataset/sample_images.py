from enum import Enum
import os
import random
import shutil

class TaskType(Enum):
    TEXTUAL_INVERSIONS = "textual_inversions"
    LORA = "lora"


classes = ['broadleaved_indigenous_hardwood', 'deciduous_hardwood', 'grose_broom', 'harvested_forest', 'herbaceous_freshwater_vege', 'high_producing_grassland', 'indigenous_forest', 'lake_pond', 'low_producing_grassland', 'manuka_kanuka', 'shortrotation_cropland', 'urban_build_up', 'urban_parkland']
task = TaskType.LORA
limit = 5


for class_name in classes:
    source_folder = f"/home/dj191/research/dataset/classification/train/{class_name}"
    destination_folder = f"/home/dj191/research/dataset/preprocessed/{task.value}/{class_name}"
    os.makedirs(destination_folder, exist_ok=True)

    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    selected_files = random.sample(all_files, limit)

    for file in selected_files:
        shutil.copy(os.path.join(source_folder, file), os.path.join(destination_folder, file))

    print(f"Successfully copied {class_name} images to the destination")

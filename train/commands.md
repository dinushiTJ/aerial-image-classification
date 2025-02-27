## SD 2.1.base -- textual inversion
# model for class 1

accelerate launch textual_inversion.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" --train_data_dir "/home/dj191/research/dataset/preprocessed/textual_inversions/broadleaved_indigenous_hardwood" --output_dir "/home/dj191/research/models/textual_inversions/bih_textual_inversions" --checkpointing_steps 500 --num_vectors 6 --placeholder_token "BIH" --initializer_token "aerial" --learnable_property "style" --resolution 512 --train_batch_size 1 --max_train_steps 3000 --learning_rate 5.0e-04 --scale_lr --lr_scheduler "constant" --enable_xformers_memory_efficient_attention --validation_prompt "A BIH aerial view" --num_validation_images 2 --validation_steps 500 --push_to_hub --report_to="wandb"

# model for class 2

accelerate launch textual_inversion.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" --train_data_dir "/home/dj191/research/dataset/preprocessed/textual_inversions/deciduous_hardwood" --output_dir "/home/dj191/research/models/textual_inversions/dhw_textual_inversions" --checkpointing_steps 500 --num_vectors 6 --placeholder_token "DHW" --initializer_token "aerial" --learnable_property "style" --resolution 512 --train_batch_size 1 --max_train_steps 3000 --learning_rate 5.0e-04 --scale_lr --lr_scheduler "constant" --enable_xformers_memory_efficient_attention --validation_prompt "A DHW aerial view" --num_validation_images 2 --validation_steps 500 --push_to_hub --report_to="wandb"

<!-- accelerate launch textual_inversion.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1" --train_data_dir "/home/dj191/research/dataset/preprocessed/deciduous_hardwood" --output_dir "/home/dj191/research/dataset/textual_inversion/dhw" --checkpointing_steps 200 --num_vectors 6 --placeholder_token "DHW" --initializer_token "aerial" --learnable_property "style" --resolution 512 --train_batch_size 1 --max_train_steps 1000 --learning_rate 5.0e-04 --scale_lr --lr_scheduler "constant"  --push_to_hub --validation_prompt "A DHW aerial view" --num_validation_images 2 --validation_steps 200  -->



# SD 2.5 base - DreamBooth + LoRA
## model for class 1

accelerate launch train_dreambooth_lora.py --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" --instance_data_dir="/home/dj191/research/dataset/preprocessed/lora/broadleaved_indigenous_hardwood" --output_dir="/home/dj191/research/models/lora/bih_lora" --instance_prompt="A BIH aerial image" --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --checkpointing_steps=300 --learning_rate=1e-4 --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=1500 --seed="0" --train_text_encoder --enable_xformers_memory_efficient_attention --push_to_hub --report_to="wandb" --validation_prompt="A BIH aerial view" --validation_epochs=10 --num_validation_images=2

## model for class 2

accelerate launch train_dreambooth_lora.py --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" --instance_data_dir="/home/dj191/research/dataset/preprocessed/lora/deciduous_hardwood" --output_dir="/home/dj191/research/models/lora/dhw_lora" --instance_prompt="A DHW aerial image" --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --checkpointing_steps=300 --learning_rate=1e-4 --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=1500 --seed="0" --train_text_encoder --enable_xformers_memory_efficient_attention --push_to_hub --report_to="wandb" --validation_prompt="A DHW aerial view" --validation_epochs=10 --num_validation_images=2



---


## SDXL -- doesn't work
# model for class 3

accelerate launch textual_inversion_sdxl.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" --train_data_dir "/home/dj191/research/dataset/preprocessed/grose_broom" --output_dir "/home/dj191/research/dataset/textual_inversion/gbm" --checkpointing_steps 200 --num_vectors 6 --placeholder_token "GBM" --initializer_token "aerial" --learnable_property "style" --resolution 512 --train_batch_size 1 --max_train_steps 1000 --learning_rate 5.0e-04 --scale_lr --lr_scheduler "constant" --validation_prompt "A GBM aerial view" --num_validation_images 2 --validation_steps 200 --mixed_precision "bf16"  --gradient_accumulation_steps 4 --save_as_full_pipeline --push_to_hub
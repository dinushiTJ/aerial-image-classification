## SD 2.1.base -- textual inversion
# model for class 1

accelerate launch textual_inversion.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" --train_data_dir "/home/dj191/research/code/train/dataset/broadleaved_indigenous_hardwood" --output_dir "/home/dj191/research/code/ti_models/bih_textual_inversion" --checkpointing_steps 500 --num_vectors 6 --placeholder_token "<BIH>" --initializer_token "aerial" --learnable_property "style" --resolution 512 --train_batch_size 1 --max_train_steps 3000 --learning_rate 5.0e-04 --scale_lr --lr_scheduler "constant" --enable_xformers_memory_efficient_attention --validation_prompt "A <BIH> aerial view" --num_validation_images 2 --validation_steps 500 --push_to_hub --report_to="wandb"

# model for class 2

accelerate launch textual_inversion.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" --train_data_dir "/home/dj191/research/code/train/dataset/deciduous_hardwood" --output_dir "/home/dj191/research/code/ti_models/dhw_textual_inversion" --checkpointing_steps 500 --num_vectors 6 --placeholder_token "<DHW>" --initializer_token "aerial" --learnable_property "style" --resolution 512 --train_batch_size 1 --max_train_steps 3000 --learning_rate 5.0e-04 --scale_lr --lr_scheduler "constant" --enable_xformers_memory_efficient_attention --validation_prompt "A <DHW> aerial view" --num_validation_images 2 --validation_steps 500 --push_to_hub --report_to="wandb"

# model for class 3

accelerate launch textual_inversion.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" --train_data_dir "/home/dj191/research/code/train/dataset/grose_broom" --output_dir "/home/dj191/research/code/ti_models/gbm_textual_inversion" --checkpointing_steps 500 --num_vectors 6 --placeholder_token "<GBM>" --initializer_token "aerial" --learnable_property "style" --resolution 512 --train_batch_size 1 --max_train_steps 3000 --learning_rate 5.0e-04 --scale_lr --lr_scheduler "constant" --enable_xformers_memory_efficient_attention --validation_prompt "A <GBM> aerial view" --num_validation_images 2 --validation_steps 500 --push_to_hub --report_to="wandb"

# model for class 4

accelerate launch textual_inversion.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" --train_data_dir "/home/dj191/research/code/train/dataset/harvested_forestharvested_forest" --output_dir "/home/dj191/research/code/ti_models/hft_textual_inversion" --checkpointing_steps 500 --num_vectors 6 --placeholder_token "<HFT>" --initializer_token "aerial" --learnable_property "style" --resolution 512 --train_batch_size 1 --max_train_steps 3000 --learning_rate 5.0e-04 --scale_lr --lr_scheduler "constant" --enable_xformers_memory_efficient_attention --validation_prompt "A <HFT> aerial view" --num_validation_images 2 --validation_steps 500 --push_to_hub --report_to="wandb"

# model for class 5

accelerate launch textual_inversion.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" --train_data_dir "/home/dj191/research/code/train/dataset/herbaceous_freshwater_vege" --output_dir "/home/dj191/research/code/ti_models/hfv_textual_inversion" --checkpointing_steps 500 --num_vectors 6 --placeholder_token "<HFV>" --initializer_token "aerial" --learnable_property "style" --resolution 512 --train_batch_size 1 --max_train_steps 3000 --learning_rate 5.0e-04 --scale_lr --lr_scheduler "constant" --enable_xformers_memory_efficient_attention --validation_prompt "A <HFV> aerial view" --num_validation_images 2 --validation_steps 500 --push_to_hub --report_to="wandb"

# model for class 6

accelerate launch textual_inversion.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" --train_data_dir "/home/dj191/research/code/train/dataset/high_producing_grassland" --output_dir "/home/dj191/research/code/ti_models/hpg_textual_inversion" --checkpointing_steps 500 --num_vectors 6 --placeholder_token "<HPG>" --initializer_token "aerial" --learnable_property "style" --resolution 512 --train_batch_size 1 --max_train_steps 3000 --learning_rate 5.0e-04 --scale_lr --lr_scheduler "constant" --enable_xformers_memory_efficient_attention --validation_prompt "A <HPG> aerial view" --num_validation_images 2 --validation_steps 500 --push_to_hub --report_to="wandb"

# model for class 7

accelerate launch textual_inversion.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" --train_data_dir "/home/dj191/research/code/train/dataset/indigenous_forest" --output_dir "/home/dj191/research/code/ti_models/ift_textual_inversion" --checkpointing_steps 500 --num_vectors 6 --placeholder_token "<IFT>" --initializer_token "aerial" --learnable_property "style" --resolution 512 --train_batch_size 1 --max_train_steps 3000 --learning_rate 5.0e-04 --scale_lr --lr_scheduler "constant" --enable_xformers_memory_efficient_attention --validation_prompt "A <IFT> aerial view" --num_validation_images 2 --validation_steps 500 --push_to_hub --report_to="wandb"

# model for class 8

accelerate launch textual_inversion.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" --train_data_dir "/home/dj191/research/code/train/dataset/lake_pond" --output_dir "/home/dj191/research/code/ti_models/lpd_textual_inversion" --checkpointing_steps 500 --num_vectors 6 --placeholder_token "<LPD>" --initializer_token "aerial" --learnable_property "style" --resolution 512 --train_batch_size 1 --max_train_steps 3000 --learning_rate 5.0e-04 --scale_lr --lr_scheduler "constant" --enable_xformers_memory_efficient_attention --validation_prompt "A <LPD> aerial view" --num_validation_images 2 --validation_steps 500 --push_to_hub --report_to="wandb"

# model for class 9

accelerate launch textual_inversion.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" --train_data_dir "/home/dj191/research/code/train/dataset/low_producing_grassland" --output_dir "/home/dj191/research/code/ti_models/lpg_textual_inversion" --checkpointing_steps 500 --num_vectors 6 --placeholder_token "<LPG>" --initializer_token "aerial" --learnable_property "style" --resolution 512 --train_batch_size 1 --max_train_steps 3000 --learning_rate 5.0e-04 --scale_lr --lr_scheduler "constant" --enable_xformers_memory_efficient_attention --validation_prompt "A <LPG> aerial view" --num_validation_images 2 --validation_steps 500 --push_to_hub --report_to="wandb"

# model for class 10

accelerate launch textual_inversion.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" --train_data_dir "/home/dj191/research/code/train/dataset/manuka_kanuka" --output_dir "/home/dj191/research/code/ti_models/mka_textual_inversion" --checkpointing_steps 500 --num_vectors 6 --placeholder_token "<MKA>" --initializer_token "aerial" --learnable_property "style" --resolution 512 --train_batch_size 1 --max_train_steps 3000 --learning_rate 5.0e-04 --scale_lr --lr_scheduler "constant" --enable_xformers_memory_efficient_attention --validation_prompt "A <MKA> aerial view" --num_validation_images 2 --validation_steps 500 --push_to_hub --report_to="wandb"

# model for class 11

accelerate launch textual_inversion.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" --train_data_dir "/home/dj191/research/code/train/dataset/shortrotation_cropland" --output_dir "/home/dj191/research/code/ti_models/src_textual_inversion" --checkpointing_steps 500 --num_vectors 6 --placeholder_token "<SRC>" --initializer_token "aerial" --learnable_property "style" --resolution 512 --train_batch_size 1 --max_train_steps 3000 --learning_rate 5.0e-04 --scale_lr --lr_scheduler "constant" --enable_xformers_memory_efficient_attention --validation_prompt "A <SRC> aerial view" --num_validation_images 2 --validation_steps 500 --push_to_hub --report_to="wandb"

# model for class 12

accelerate launch textual_inversion.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" --train_data_dir "/home/dj191/research/code/train/dataset/urban_build_up" --output_dir "/home/dj191/research/code/ti_models/ubu_textual_inversion" --checkpointing_steps 500 --num_vectors 6 --placeholder_token "<UBU>" --initializer_token "aerial" --learnable_property "style" --resolution 512 --train_batch_size 1 --max_train_steps 3000 --learning_rate 5.0e-04 --scale_lr --lr_scheduler "constant" --enable_xformers_memory_efficient_attention --validation_prompt "A <UBU> aerial view" --num_validation_images 2 --validation_steps 500 --push_to_hub --report_to="wandb"

# model for class 13

accelerate launch textual_inversion.py --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" --train_data_dir "/home/dj191/research/code/train/dataset/urban_parkland" --output_dir "/home/dj191/research/code/ti_models/upl_textual_inversion" --checkpointing_steps 500 --num_vectors 6 --placeholder_token "<UPL>" --initializer_token "aerial" --learnable_property "style" --resolution 512 --train_batch_size 1 --max_train_steps 3000 --learning_rate 5.0e-04 --scale_lr --lr_scheduler "constant" --enable_xformers_memory_efficient_attention --validation_prompt "A <UPL> aerial view" --num_validation_images 2 --validation_steps 500 --push_to_hub --report_to="wandb"

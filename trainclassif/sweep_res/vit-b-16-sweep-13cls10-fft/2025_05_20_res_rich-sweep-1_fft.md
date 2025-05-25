
🏁 Training Summary
--------------------
```json
"{'augment_level': 'medium', 'batch_size': 32, 'calc_dataset_stats': False, 'cutmix_alpha': 1, 'cutmix_or_mixup': True, 'dataset_name': 'dushj98/aerial_real_plus_0010', 'dropout_p': 0.19763987719505707, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0.1, 'layers_to_unfreeze': ['encoder.layers.encoder_layer_8.', 'encoder.layers.encoder_layer_9.', 'encoder.layers.encoder_layer_10.', 'encoder.layers.encoder_layer_11.'], 'learning_rate': 3.995396750112999e-06, 'mixup_alpha': 0.2, 'model': 'vit_b_16', 'patience': 10, 'scheduler': ['cosine', 'plateau'], 'seed': 42, 'training_mode': 'fft', 'use_mixed_precision': True, 'weight_decay': 0.011018642173028706, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 13, 'data_normalization_strategy': 'ImageNet', 'dataset_mean': None, 'dataset_std': None, 'optimizer': 'AdamW'}"


| Model    | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|----------|----------------|--------------|------------------|------------|
| vit_b_16 | 50.58%         |           14 |               21 | ✅ Success |

🏁 Training Summary
--------------------
```json
"{'augment_level': 'basic', 'batch_size': 32, 'calc_dataset_stats': False, 'cutmix_alpha': 1, 'cutmix_or_mixup': True, 'dataset_name': 'dushj98/aerial_real_only', 'dropout_p': 0.15476565592540073, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0.2, 'layers_to_unfreeze': ['encoder.layers.encoder_layer_8.', 'encoder.layers.encoder_layer_9.', 'encoder.layers.encoder_layer_10.', 'encoder.layers.encoder_layer_11.'], 'learning_rate': 0.0004473400911727381, 'mixup_alpha': 0.2, 'model': 'vit_b_16', 'patience': 10, 'scheduler': 'plateau', 'seed': 42, 'training_mode': 'tl', 'use_mixed_precision': True, 'weight_decay': 0.0737428836225661, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 13, 'data_normalization_strategy': 'ImageNet', 'dataset_mean': None, 'dataset_std': None, 'optimizer': 'Adam'}"


| Model    | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|----------|----------------|--------------|------------------|------------|
| vit_b_16 | 46.43%         |           24 |             25.4 | ✅ Success |
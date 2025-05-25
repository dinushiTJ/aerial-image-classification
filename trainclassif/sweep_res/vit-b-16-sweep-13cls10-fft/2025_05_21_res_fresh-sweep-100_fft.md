
🏁 Training Summary
--------------------
```json
"{'augment_level': 'none', 'batch_size': 32, 'calc_dataset_stats': False, 'cutmix_alpha': 1, 'cutmix_or_mixup': True, 'dataset_name': 'dushj98/aerial_real_plus_0010', 'dropout_p': 0.48799698985859485, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0.2, 'layers_to_unfreeze': ['encoder.layers.encoder_layer_8.', 'encoder.layers.encoder_layer_9.', 'encoder.layers.encoder_layer_10.', 'encoder.layers.encoder_layer_11.'], 'learning_rate': 3.1378015724580075e-05, 'mixup_alpha': 0.2, 'model': 'vit_b_16', 'patience': 10, 'scheduler': ['cosine', 'plateau'], 'seed': 42, 'training_mode': 'fft', 'use_mixed_precision': True, 'weight_decay': 0.02407886910127361, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 13, 'data_normalization_strategy': 'ImageNet', 'dataset_mean': None, 'dataset_std': None, 'optimizer': 'AdamW'}"


| Model    | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|----------|----------------|--------------|------------------|------------|
| vit_b_16 | 52.46%         |            3 |             11.6 | ✅ Success |

🏁 Training Summary
--------------------
```json
"{'augment_level': 'none', 'batch_size': 32, 'calc_dataset_stats': False, 'cutmix_alpha': 1, 'cutmix_or_mixup': True, 'dataset_name': 'dushj98/aerial_real_plus_0010', 'dropout_p': 0.31984293362449123, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0.1, 'layers_to_unfreeze': ['encoder.layers.encoder_layer_8.', 'encoder.layers.encoder_layer_9.', 'encoder.layers.encoder_layer_10.', 'encoder.layers.encoder_layer_11.'], 'learning_rate': 1.0298156877974382e-05, 'mixup_alpha': 0.2, 'model': 'vit_b_16', 'patience': 10, 'scheduler': ['cosine', 'plateau'], 'seed': 42, 'training_mode': 'fft', 'use_mixed_precision': True, 'weight_decay': 0.010065959596540393, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 13, 'data_normalization_strategy': 'ImageNet', 'dataset_mean': None, 'dataset_std': None, 'optimizer': 'AdamW'}"


| Model    | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|----------|----------------|--------------|------------------|------------|
| vit_b_16 | 52.35%         |            5 |             13.3 | ✅ Success |

🏁 Training Summary
--------------------
```json
"{'augment_level': 'basic', 'batch_size': 8, 'calc_dataset_stats': False, 'dataset_name': 'dushj98/aerial_real_only', 'dropout_p': 0.10297066102018972, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0.2, 'layers_to_unfreeze': ['encoder.layers.encoder_layer_8.', 'encoder.layers.encoder_layer_9.', 'encoder.layers.encoder_layer_10.', 'encoder.layers.encoder_layer_11.'], 'learning_rate': 1.3515749937796084e-05, 'model': 'vit_b_16', 'patience': 10, 'scheduler': 'cosine', 'seed': 42, 'training_mode': 'fft', 'use_mixed_precision': True, 'weight_decay': 0.013689148598447146, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 13, 'data_normalization_strategy': 'ImageNet', 'dataset_mean': None, 'dataset_std': None, 'optimizer': 'AdamW'}"


| Model    | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|----------|----------------|--------------|------------------|------------|
| vit_b_16 | 50.55%         |            2 |             17.3 | ✅ Success |
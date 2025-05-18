
🏁 Training Summary
--------------------
```json
"{'augment_level': 'basic', 'batch_size': 32, 'calc_dataset_stats': False, 'dataset_name': 'dushj98/waikato_aerial_imagery_2017_7cls', 'dropout_p': 0.3779780670326388, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0.1, 'layers_to_unfreeze': ['features.6', 'features.7', 'features.8'], 'learning_rate': 0.00010189795492701968, 'model': 'efficientnetv2_m', 'patience': 10, 'scheduler': 'plateau', 'seed': 42, 'training_mode': 'fft', 'use_mixed_precision': True, 'weight_decay': 0.0705538756248657, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 7, 'data_normalization_strategy': 'ImageNet', 'dataset_mean': None, 'dataset_std': None, 'optimizer': 'AdamW'}"


| Model            | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|------------------|----------------|--------------|------------------|------------|
| efficientnetv2_m | 74.12%         |           21 |             13.5 | ✅ Success |
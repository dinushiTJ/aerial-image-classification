
🏁 Training Summary
--------------------
```json
"{'augment_level': 'basic', 'batch_size': 32, 'calc_dataset_stats': False, 'dataset_name': 'dushj98/waikato_aerial_imagery_2017_7cls', 'dropout_p': 0.27319647693576377, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0.1, 'layers_to_unfreeze': ['features.6', 'features.7', 'features.8'], 'learning_rate': 0.0003029333748463635, 'model': 'efficientnetv2_m', 'patience': 10, 'scheduler': 'cosine', 'seed': 42, 'training_mode': 'fft', 'use_mixed_precision': True, 'weight_decay': 0.020605572509050273, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 7, 'data_normalization_strategy': 'ImageNet', 'dataset_mean': None, 'dataset_std': None, 'optimizer': 'AdamW'}"


| Model            | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|------------------|----------------|--------------|------------------|------------|
| efficientnetv2_m | 75.36%         |           15 |             11.8 | ✅ Success |
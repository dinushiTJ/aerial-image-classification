
🏁 Training Summary
--------------------
```json
"{'augment_level': 'medium', 'batch_size': 32, 'calc_dataset_stats': False, 'dataset_name': 'dushj98/waikato_aerial_imagery_2017_7cls', 'dropout_p': 0.48778098522274593, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0.2, 'layers_to_unfreeze': ['features.6', 'features.7', 'features.8'], 'learning_rate': 7.409972269874855e-06, 'model': 'efficientnetv2_m', 'patience': 10, 'scheduler': 'plateau', 'seed': 42, 'training_mode': 'sft', 'use_mixed_precision': True, 'weight_decay': 0.1666400110650288, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 7, 'data_normalization_strategy': 'ImageNet', 'dataset_mean': None, 'dataset_std': None, 'optimizer': 'AdamW'}"


| Model            | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|------------------|----------------|--------------|------------------|------------|
| efficientnetv2_m | 69.12%         |           18 |             10.9 | ✅ Success |

🏁 Training Summary
--------------------
```json
"{'augment_level': 'none', 'batch_size': 16, 'calc_dataset_stats': False, 'dataset_name': 'dushj98/waikato_aerial_imagery_2017_7cls', 'dropout_p': 0.26275717258687625, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0.2, 'layers_to_unfreeze': ['features.14.', 'features.15.', 'features.16.', 'features.17.', 'features.18.'], 'learning_rate': 0.0004030612645841971, 'model': 'mobilenet_v2', 'patience': 10, 'scheduler': 'plateau', 'seed': 42, 'training_mode': 'tl', 'use_mixed_precision': True, 'weight_decay': 0.15236040255398547, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 7, 'data_normalization_strategy': 'ImageNet', 'dataset_mean': None, 'dataset_std': None, 'optimizer': 'Adam'}"


| Model        | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|--------------|----------------|--------------|------------------|------------|
| mobilenet_v2 | 64.24%         |           24 |             12.2 | ✅ Success |
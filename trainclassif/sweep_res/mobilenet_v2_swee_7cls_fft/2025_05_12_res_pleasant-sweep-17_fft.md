
🏁 Training Summary
--------------------
```json
"{'augment_level': 'medium', 'batch_size': 64, 'calc_dataset_stats': False, 'dataset_name': 'dushj98/waikato_aerial_imagery_2017_7cls', 'dropout_p': 0.16025705932529957, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0.2, 'layers_to_unfreeze': ['features.14.', 'features.15.', 'features.16.', 'features.17.', 'features.18.'], 'learning_rate': 0.0004114112964558935, 'model': 'mobilenet_v2', 'patience': 10, 'scheduler': 'cosine', 'seed': 42, 'training_mode': 'fft', 'use_mixed_precision': True, 'weight_decay': 0.1979539043063397, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 7, 'data_normalization_strategy': 'ImageNet', 'dataset_mean': None, 'dataset_std': None, 'optimizer': 'AdamW'}"


| Model        | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|--------------|----------------|--------------|------------------|------------|
| mobilenet_v2 | 72.97%         |            8 |                8 | ✅ Success |
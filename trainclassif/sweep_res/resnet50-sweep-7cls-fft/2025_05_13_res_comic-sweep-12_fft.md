
🏁 Training Summary
--------------------
```json
"{'augment_level': 'basic', 'batch_size': 64, 'calc_dataset_stats': False, 'dataset_name': 'dushj98/waikato_aerial_imagery_2017_7cls', 'dropout_p': 0.2986811334539814, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0, 'layers_to_unfreeze': ['layer4.'], 'learning_rate': 4.697760541550095e-06, 'model': 'resnet50', 'patience': 10, 'scheduler': 'none', 'seed': 42, 'training_mode': 'fft', 'use_mixed_precision': True, 'weight_decay': 0.03784606388150257, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 7, 'data_normalization_strategy': 'ImageNet', 'dataset_mean': None, 'dataset_std': None, 'optimizer': 'AdamW'}"


| Model    | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|----------|----------------|--------------|------------------|------------|
| resnet50 | 69.63%         |           23 |             11.4 | ✅ Success |
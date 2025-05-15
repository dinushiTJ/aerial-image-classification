
🏁 Training Summary
--------------------
```json
"{'augment_level': 'basic', 'batch_size': 64, 'calc_dataset_stats': False, 'dataset_name': 'dushj98/waikato_aerial_imagery_2017_7cls', 'dropout_p': 0.4425537080701813, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0.1, 'layers_to_unfreeze': ['layer4.'], 'learning_rate': 0.0001874594290961546, 'model': 'resnet50', 'patience': 10, 'scheduler': 'plateau', 'seed': 42, 'training_mode': 'fft', 'use_mixed_precision': True, 'weight_decay': 0.19600378310049785, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 7, 'data_normalization_strategy': 'ImageNet', 'dataset_mean': None, 'dataset_std': None, 'optimizer': 'AdamW'}"


| Model    | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|----------|----------------|--------------|------------------|------------|
| resnet50 | 72.50%         |            4 |              5.8 | ✅ Success |
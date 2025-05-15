
🏁 Training Summary
--------------------
```json
"{'augment_level': 'basic', 'batch_size': 64, 'calc_dataset_stats': False, 'dataset_name': 'dushj98/waikato_aerial_imagery_2017_7cls', 'dropout_p': 0.5807442012590773, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0, 'layers_to_unfreeze': ['layer4.'], 'learning_rate': 7.391468589481161e-06, 'model': 'resnet50', 'patience': 10, 'scheduler': 'cosine', 'seed': 42, 'training_mode': 'fft', 'use_mixed_precision': True, 'weight_decay': 0.07115298054379404, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 7, 'data_normalization_strategy': 'ImageNet', 'dataset_mean': None, 'dataset_std': None, 'optimizer': 'AdamW'}"


| Model    | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|----------|----------------|--------------|------------------|------------|
| resnet50 | 68.95%         |           22 |             11.3 | ✅ Success |
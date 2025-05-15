
🏁 Training Summary
--------------------
```json
"{'augment_level': 'basic', 'batch_size': 64, 'calc_dataset_stats': False, 'dataset_name': 'dushj98/waikato_aerial_imagery_2017_7cls', 'dropout_p': 0.18912445236316025, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0.2, 'layers_to_unfreeze': ['layer4.'], 'learning_rate': 6.387366998269587e-05, 'model': 'resnet50', 'patience': 10, 'scheduler': 'cosine', 'seed': 42, 'training_mode': 'sft', 'use_mixed_precision': True, 'weight_decay': 0.14440314810692412, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 7, 'data_normalization_strategy': 'ImageNet', 'dataset_mean': None, 'dataset_std': None, 'optimizer': 'AdamW'}"


| Model    | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|----------|----------------|--------------|------------------|------------|
| resnet50 | 70.44%         |           18 |               11 | ✅ Success |
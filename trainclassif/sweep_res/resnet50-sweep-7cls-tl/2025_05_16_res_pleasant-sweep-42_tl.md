
🏁 Training Summary
--------------------
```json
"{'augment_level': 'none', 'batch_size': 64, 'calc_dataset_stats': False, 'dataset_name': 'dushj98/waikato_aerial_imagery_2017_7cls', 'dropout_p': 0.12693456017595653, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0, 'layers_to_unfreeze': ['layer4.'], 'learning_rate': 0.00044094355920760607, 'model': 'resnet50', 'patience': 10, 'scheduler': 'none', 'seed': 42, 'training_mode': 'tl', 'use_mixed_precision': True, 'weight_decay': 0.1023778406830754, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 7, 'data_normalization_strategy': 'ImageNet', 'dataset_mean': None, 'dataset_std': None, 'optimizer': 'Adam'}"


| Model    | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|----------|----------------|--------------|------------------|------------|
| resnet50 | 65.18%         |           19 |             11.8 | ✅ Success |
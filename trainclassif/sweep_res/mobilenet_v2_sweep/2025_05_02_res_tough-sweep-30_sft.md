
🏁 Training Summary
--------------------
```json
"{'augment_level': 'medium', 'batch_size': 64, 'calc_dataset_stats': True, 'dataset_name': 'dushj98/aerial_real_only', 'dropout_p': 0.33599572508672226, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0, 'layers_to_unfreeze': ['features.14.', 'features.15.', 'features.16.', 'features.17.', 'features.18.'], 'learning_rate': 0.00010423168084193558, 'model': 'mobilenet_v2', 'patience': 10, 'save_to_artifacts': False, 'scheduler': 'cosine', 'seed': 42, 'training_mode': 'sft', 'use_mixed_precision': True, 'weight_decay': 0.014745216914985754, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 13, 'data_normalization_strategy': 'dataset', 'dataset_mean': [0.3451549708843231, 0.4062410295009613, 0.35439378023147583], 'dataset_std': [0.1031293198466301, 0.0795460194349289, 0.06038741022348404], 'optimizer': 'AdamW'}"


| Model        | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|--------------|----------------|--------------|------------------|------------|
| mobilenet_v2 | 45.60%         |            5 |             10.5 | ✅ Success |
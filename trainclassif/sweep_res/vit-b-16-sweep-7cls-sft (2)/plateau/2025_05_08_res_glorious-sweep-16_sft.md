
🏁 Training Summary
--------------------
```json
"{'augment_level': 'basic', 'batch_size': 16, 'calc_dataset_stats': False, 'dataset_name': 'dushj98/waikato_aerial_imagery_2017_7cls', 'dropout_p': 0.2913524959763175, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0, 'layers_to_unfreeze': ['encoder.layers.encoder_layer_8.', 'encoder.layers.encoder_layer_9.', 'encoder.layers.encoder_layer_10.', 'encoder.layers.encoder_layer_11.'], 'learning_rate': 8.849565257657589e-05, 'model': 'vit_b_16', 'patience': 10, 'scheduler': 'plateau', 'seed': 42, 'training_mode': 'sft', 'use_mixed_precision': True, 'weight_decay': 0.08331053657829099, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 7, 'data_normalization_strategy': 'ImageNet', 'dataset_mean': None, 'dataset_std': None, 'optimizer': 'AdamW'}"


| Model    | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|----------|----------------|--------------|------------------|------------|
| vit_b_16 | 73.61%         |            2 |              4.5 | ✅ Success |
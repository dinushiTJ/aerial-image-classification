
🏁 Training Summary
--------------------
```json
"{'augment_level': 'none', 'batch_size': 8, 'calc_dataset_stats': False, 'dataset_name': 'dushj98/waikato_aerial_imagery_2017_7cls', 'dropout_p': 0.32700890878394273, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0, 'layers_to_unfreeze': ['encoder.layers.encoder_layer_8.', 'encoder.layers.encoder_layer_9.', 'encoder.layers.encoder_layer_10.', 'encoder.layers.encoder_layer_11.'], 'learning_rate': 0.0004802348128421664, 'model': 'vit_b_16', 'patience': 10, 'scheduler': 'plateau', 'seed': 42, 'training_mode': 'tl', 'use_mixed_precision': True, 'weight_decay': 0.05425370673308608, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 7, 'data_normalization_strategy': 'ImageNet', 'dataset_mean': None, 'dataset_std': None, 'optimizer': 'Adam'}"


| Model    | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|----------|----------------|--------------|------------------|------------|
| vit_b_16 | 68.22%         |           29 |             10.6 | ✅ Success |
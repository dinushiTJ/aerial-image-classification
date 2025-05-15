
🏁 Training Summary
--------------------
```json
"{'augment_level': 'none', 'batch_size': 16, 'calc_dataset_stats': False, 'dataset_name': 'dushj98/waikato_aerial_imagery_2017_8cls', 'dropout_p': 0.15290222700958586, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0.1, 'layers_to_unfreeze': ['encoder.layers.encoder_layer_8.', 'encoder.layers.encoder_layer_9.', 'encoder.layers.encoder_layer_10.', 'encoder.layers.encoder_layer_11.'], 'learning_rate': 0.00039212795509496057, 'model': 'vit_b_16', 'patience': 10, 'scheduler': 'cosine', 'seed': 42, 'training_mode': 'tl', 'use_mixed_precision': True, 'weight_decay': 0.03886846295715167, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 8, 'data_normalization_strategy': 'ImageNet', 'dataset_mean': None, 'dataset_std': None, 'optimizer': 'Adam'}"


| Model    | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|----------|----------------|--------------|------------------|------------|
| vit_b_16 | 60.14%         |           19 |             12.2 | ✅ Success |
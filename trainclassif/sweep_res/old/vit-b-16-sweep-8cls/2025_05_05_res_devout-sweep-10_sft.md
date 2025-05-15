
🏁 Training Summary
--------------------
```json
"{'augment_level': 'basic', 'batch_size': 16, 'calc_dataset_stats': False, 'dataset_name': 'dushj98/waikato_aerial_imagery_2017_8cls', 'dropout_p': 0.2304209964410289, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0.2, 'layers_to_unfreeze': ['encoder.layers.encoder_layer_8.', 'encoder.layers.encoder_layer_9.', 'encoder.layers.encoder_layer_10.', 'encoder.layers.encoder_layer_11.'], 'learning_rate': 5.285469114589498e-05, 'model': 'vit_b_16', 'patience': 10, 'scheduler': 'cosine', 'seed': 42, 'training_mode': 'sft', 'use_mixed_precision': True, 'weight_decay': 0.04352714320133732, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 8, 'data_normalization_strategy': 'ImageNet', 'dataset_mean': None, 'dataset_std': None, 'optimizer': 'AdamW'}"


| Model    | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|----------|----------------|--------------|------------------|------------|
| vit_b_16 | 65.49%         |            9 |                8 | ✅ Success |
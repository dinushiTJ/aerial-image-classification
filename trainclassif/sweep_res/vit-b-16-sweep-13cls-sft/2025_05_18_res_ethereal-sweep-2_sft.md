
🏁 Training Summary
--------------------
```json
"{'augment_level': 'basic', 'batch_size': 32, 'calc_dataset_stats': False, 'cutmix_alpha': 1, 'cutmix_or_mixup': False, 'dataset_name': 'dushj98/aerial_real_only', 'dropout_p': 0.12199361874031198, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0.1, 'layers_to_unfreeze': ['encoder.layers.encoder_layer_8.', 'encoder.layers.encoder_layer_9.', 'encoder.layers.encoder_layer_10.', 'encoder.layers.encoder_layer_11.'], 'learning_rate': 0.00011024435438685814, 'mixup_alpha': 0.2, 'model': 'vit_b_16', 'patience': 10, 'scheduler': 'cosine', 'seed': 42, 'training_mode': 'sft', 'use_mixed_precision': True, 'weight_decay': 0.020144154793708936, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 13, 'data_normalization_strategy': 'ImageNet', 'dataset_mean': None, 'dataset_std': None, 'optimizer': 'AdamW'}"


| Model    | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|----------|----------------|--------------|------------------|------------|
| vit_b_16 | 49.40%         |           14 |             17.1 | ✅ Success |

🏁 Training Summary
--------------------
```json
"{'augment_level': 'medium', 'batch_size': 32, 'calc_dataset_stats': True, 'dataset_name': 'dushj98/aerial_real_only', 'dropout_p': 0.4577696398954883, 'epochs': 30, 'input_size': 224, 'label_smoothing': 0.2, 'layers_to_unfreeze': ['encoder.layers.encoder_layer_8.', 'encoder.layers.encoder_layer_9.', 'encoder.layers.encoder_layer_10.', 'encoder.layers.encoder_layer_11.'], 'learning_rate': 0.0001696406657170987, 'model': 'vit_b_16', 'patience': 10, 'scheduler': 'cosine', 'seed': 42, 'training_mode': 'sft', 'use_mixed_precision': True, 'weight_decay': 0.01591519355685982, 'actual_use_mixed_precision': True, 'cuda': True, 'num_classes': 13, 'data_normalization_strategy': 'dataset', 'dataset_mean': [0.3451549708843231, 0.4062410295009613, 0.35439378023147583], 'dataset_std': [0.1031293198466301, 0.0795460194349289, 0.06038741022348404], 'optimizer': 'AdamW'}"


| Model    | Best Val Acc   |   Best Epoch |   Train Time (m) | Status     |
|----------|----------------|--------------|------------------|------------|
| vit_b_16 | 49.26%         |            2 |              8.3 | ✅ Success |
| field                       | tl                 | sft                | fft                |
|-----------------------------|--------------------|--------------------|--------------------|
| training_mode               | tl                 | sft                | fft                |
| best_val_acc                | 0.3935974205435283 | 0.5032243205895901 | 0.518885306310456  |
| dropout_p                   | 0.1                | 0.4                | 0.3                |
| learning_rate               | 0.0002             | 4e-05              | 1e-05              |
| label_smoothing             | 0.0                | 0.2                | 0.2                |
| weight_decay                | 0.02               | 0.14               | 0.03               |
| augment_level               | none               | none               | none               |
| scheduler                   | plateau            | cosine             | plateau            |
| batch_size                  | 32                 | 32                 | 32                 |
| cutmix_or_mixup             | True               | True               | True               |
| cutmix_alpha                | 1                  | 1                  | 1                  |
| mixup_alpha                 | 0.2                | 0.2                | 0.2                |
| best_epoch                  | 20                 | 14                 | 29                 |
| actual_use_mixed_precision  | True               | True               | True               |
| data_normalization_strategy | ImageNet           | ImageNet           | ImageNet           |
| epochs                      | 50                 | 50                 | 50                 |
| num_classes                 | 13                 | 13                 | 13                 |
| model                       | efficientnet_b2    | efficientnet_b2    | efficientnet_b2    |
| optimizer                   | Adam               | AdamW              | AdamW              |
| patience                    | 10                 | 10                 | 10                 |
| acc/train                   | 0                  | 0                  | 0                  |
| acc/val                     | 0.3885306310456011 | 0.488945186549977  | 0.5156609857208659 |
| train_acc_at_best_epoch     |                    |                    |                    |
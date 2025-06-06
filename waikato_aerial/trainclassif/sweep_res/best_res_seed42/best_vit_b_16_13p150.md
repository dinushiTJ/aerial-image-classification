| field                       | tl                 | sft                | fft                 |
|-----------------------------|--------------------|--------------------|---------------------|
| training_mode               | tl                 | sft                | fft                 |
| best_val_acc                | 0.4288346384154767 | 0.5                | 0.5218793182865039  |
| dropout_p                   | 0.2                | 0.4                | 0.5                 |
| learning_rate               | 0.0005             | 0.0001             | 5e-06               |
| label_smoothing             | 0.0                | 0.2                | 0.0                 |
| weight_decay                | 0.13               | 0.06               | 0.06                |
| augment_level               | none               | none               | none                |
| scheduler                   | cosine             | cosine             | plateau             |
| batch_size                  | 32                 | 32                 | 32                  |
| cutmix_or_mixup             | True               | True               | True                |
| cutmix_alpha                | 1                  | 1                  | 1                   |
| mixup_alpha                 | 0.2                | 0.2                | 0.2                 |
| best_epoch                  | 19                 | 3                  | 8                   |
| actual_use_mixed_precision  | True               | True               | True                |
| data_normalization_strategy | ImageNet           | ImageNet           | ImageNet            |
| epochs                      | 50                 | 50                 | 50                  |
| num_classes                 | 13                 | 13                 | 13                  |
| model                       | vit_b_16           | vit_b_16           | vit_b_16            |
| optimizer                   | Adam               | AdamW              | AdamW               |
| patience                    | 10                 | 15                 | 15                  |
| acc/train                   | 0                  | 0                  | 0                   |
| acc/val                     | 0.4288346384154767 | 0.4836480884385076 | 0.49723629663749425 |
| train_acc_at_best_epoch     |                    |                    |                     |
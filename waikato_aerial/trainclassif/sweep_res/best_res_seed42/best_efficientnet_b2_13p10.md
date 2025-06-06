| field                       | tl                 | sft                 | fft                |
|-----------------------------|--------------------|---------------------|--------------------|
| training_mode               | tl                 | sft                 | fft                |
| best_val_acc                | 0.4037309995393828 | 0.5122063565177337  | 0.5255642561031783 |
| dropout_p                   | 0.1                | 0.4                 | 0.3                |
| learning_rate               | 0.0002             | 4e-05               | 1e-05              |
| label_smoothing             | 0.0                | 0.2                 | 0.2                |
| weight_decay                | 0.02               | 0.14                | 0.03               |
| augment_level               | none               | none                | none               |
| scheduler                   | plateau            | cosine              | plateau            |
| batch_size                  | 32                 | 32                  | 32                 |
| cutmix_or_mixup             | True               | True                | True               |
| cutmix_alpha                | 1                  | 1                   | 1                  |
| mixup_alpha                 | 0.2                | 0.2                 | 0.2                |
| best_epoch                  | 21                 | 20                  | 43                 |
| actual_use_mixed_precision  | True               | True                | True               |
| data_normalization_strategy | ImageNet           | ImageNet            | ImageNet           |
| epochs                      | 50                 | 50                  | 50                 |
| num_classes                 | 13                 | 13                  | 13                 |
| model                       | efficientnet_b2    | efficientnet_b2     | efficientnet_b2    |
| optimizer                   | Adam               | AdamW               | AdamW              |
| patience                    | 10                 | 10                  | 10                 |
| acc/train                   | 0                  | 0                   | 0                  |
| acc/val                     | 0.3970520497466606 | 0.49078765545831415 | 0.5186549976969138 |
| train_acc_at_best_epoch     |                    |                     |                    |
| field                       | tl                  | sft                 | fft                |
|-----------------------------|---------------------|---------------------|--------------------|
| training_mode               | tl                  | sft                 | fft                |
| best_val_acc                | 0.45900506678949793 | 0.5059880239520959  | 0.5255642561031783 |
| dropout_p                   | 0.2                 | 0.4                 | 0.5                |
| learning_rate               | 0.0005              | 0.0001              | 5e-06              |
| label_smoothing             | 0.0                 | 0.2                 | 0.0                |
| weight_decay                | 0.13                | 0.06                | 0.06               |
| augment_level               | none                | none                | none               |
| scheduler                   | cosine              | cosine              | plateau            |
| batch_size                  | 32                  | 32                  | 32                 |
| cutmix_or_mixup             | True                | True                | True               |
| cutmix_alpha                | 1                   | 1                   | 1                  |
| mixup_alpha                 | 0.2                 | 0.2                 | 0.2                |
| best_epoch                  | 36                  | 3                   | 10                 |
| actual_use_mixed_precision  | True                | True                | True               |
| data_normalization_strategy | ImageNet            | ImageNet            | ImageNet           |
| epochs                      | 50                  | 50                  | 50                 |
| num_classes                 | 13                  | 13                  | 13                 |
| model                       | vit_b_16            | vit_b_16            | vit_b_16           |
| optimizer                   | Adam                | AdamW               | AdamW              |
| patience                    | 10                  | 10                  | 15                 |
| acc/train                   | 0                   | 0                   | 0                  |
| acc/val                     | 0.4571625978811608  | 0.47650852141870104 | 0.5046061722708429 |
| train_acc_at_best_epoch     |                     |                     |                    |
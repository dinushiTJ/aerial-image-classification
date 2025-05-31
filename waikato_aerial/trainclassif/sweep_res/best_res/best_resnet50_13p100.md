| field                       | tl                 | sft                | fft                 |
|-----------------------------|--------------------|--------------------|---------------------|
| training_mode               | tl                 | sft                | fft                 |
| best_val_acc                | 0.4062643942883464 | 0.4788116075541225 | 0.49792722247812066 |
| dropout_p                   | 0.5                | 0.2                | 0.4                 |
| learning_rate               | 0.0003             | 7e-06              | 7e-06               |
| label_smoothing             | 0.2                | 0.0                | 0.0                 |
| weight_decay                | 0.02               | 0.19               | 0.04                |
| augment_level               | none               | none               | none                |
| scheduler                   | plateau            | plateau            | plateau             |
| batch_size                  | 32                 | 32                 | 32                  |
| cutmix_or_mixup             | True               | True               | True                |
| cutmix_alpha                | 1                  | 1                  | 1                   |
| mixup_alpha                 | 0.2                | 0.2                | 0.2                 |
| best_epoch                  | 22                 | 44                 | 27                  |
| actual_use_mixed_precision  | True               | True               | True                |
| data_normalization_strategy | ImageNet           | ImageNet           | ImageNet            |
| epochs                      | 50                 | 50                 | 50                  |
| num_classes                 | 13                 | 13                 | 13                  |
| model                       | resnet50           | resnet50           | resnet50            |
| optimizer                   | Adam               | AdamW              | AdamW               |
| patience                    | 10                 | 10                 | 10                  |
| acc/train                   | 0                  | 0                  | 0                   |
| acc/val                     | 0.4011976047904192 | 0.4737448180561953 | 0.4880239520958084  |
| train_acc_at_best_epoch     |                    |                    |                     |
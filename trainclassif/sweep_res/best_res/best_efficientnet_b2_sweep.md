| field                       | tl                    | sft                   | fft                    |
|-----------------------------|-----------------------|-----------------------|------------------------|
| training_mode               | tl                    | sft                   | fft                    |
| best_val_acc                | 0.4062643942883464    | 0.5147397512666974    | 0.5131275909719023     |
| dropout_p                   | 0.12473320627750992   | 0.3615446264779709    | 0.28663765276170183    |
| learning_rate               | 0.0002442828934817103 | 3.825683660340811e-05 | 1.0883894132768148e-05 |
| label_smoothing             | 0.0                   | 0.2                   | 0.2                    |
| weight_decay                | 0.02365284506576477   | 0.1374004551867758    | 0.02917168099201588    |
| augment_level               | none                  | basic                 | none                   |
| scheduler                   | plateau               | cosine                | plateau                |
| batch_size                  | 32                    | 32                    | 32                     |
| cutmix_or_mixup             | True                  | True                  | True                   |
| cutmix_alpha                | 1                     | 1                     | 1                      |
| mixup_alpha                 | 0.2                   | 0.2                   | 0.2                    |
| best_epoch                  | 26                    | 24                    | 27                     |
| actual_use_mixed_precision  | True                  | True                  | True                   |
| data_normalization_strategy | ImageNet              | ImageNet              | ImageNet               |
| epochs                      | 30                    | 30                    | 30                     |
| num_classes                 | 13                    | 13                    | 13                     |
| model                       | efficientnet_b2       | efficientnet_b2       | efficientnet_b2        |
| optimizer                   | Adam                  | AdamW                 | AdamW                  |
| patience                    | 10                    | 10                    | 10                     |
| acc/train                   | 0                     | 0                     | 0                      |
| acc/val                     | 0.4058037770612621    | 0.5080608014739751    | 0.5105941962229388     |
| train_acc_at_best_epoch     |                       |                       |                        |
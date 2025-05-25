| field                       | tl                     | sft                    | fft                    |
|-----------------------------|------------------------|------------------------|------------------------|
| training_mode               | tl                     | sft                    | fft                    |
| best_val_acc                | 0.6839178785286569     | 0.7510692899914457     | 0.7634730538922155     |
| dropout_p                   | 0.2019394631142922     | 0.3069406239878064     | 0.4805742702759015     |
| learning_rate               | 0.00042859821052889567 | 0.00018972961760615816 | 2.8741637729239332e-05 |
| label_smoothing             | 0.2                    | 0.1                    | 0.1                    |
| weight_decay                | 0.054073413758165066   | 0.03293282606791158    | 0.02774603576253194    |
| augment_level               | none                   | basic                  | none                   |
| scheduler                   | plateau                | plateau                | plateau                |
| batch_size                  | 16                     | 16                     | 32                     |
| cutmix_or_mixup             |                        |                        |                        |
| cutmix_alpha                |                        |                        |                        |
| mixup_alpha                 |                        |                        |                        |
| best_epoch                  | 25                     | 22                     | 9                      |
| actual_use_mixed_precision  | True                   | True                   | True                   |
| data_normalization_strategy | ImageNet               | ImageNet               | ImageNet               |
| epochs                      | 30                     | 30                     | 30                     |
| num_classes                 | 7                      | 7                      | 7                      |
| model                       | vit_b_16               | vit_b_16               | vit_b_16               |
| optimizer                   | Adam                   | AdamW                  | AdamW                  |
| patience                    | 10                     | 10                     | 10                     |
| acc/train                   | 0.7616902616902617     | 0.9987129987129988     | 0.9995709995709996     |
| acc/val                     | 0.6770744225834047     | 0.7476475620188195     | 0.7549187339606501     |
| train_acc_at_best_epoch     |                        |                        |                        |
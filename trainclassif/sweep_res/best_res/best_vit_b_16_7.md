| field                       | sft                    | tl                     | fft                    |
|-----------------------------|------------------------|------------------------|------------------------|
| actual_use_mixed_precision  | True                   | True                   | True                   |
| augment_level               | basic                  | none                   | none                   |
| batch_size                  | 16                     | 16                     | 32                     |
| data_normalization_strategy | ImageNet               | ImageNet               | ImageNet               |
| dropout_p                   | 0.3069406239878064     | 0.2019394631142922     | 0.4805742702759015     |
| epochs                      | 30                     | 30                     | 30                     |
| label_smoothing             | 0.1                    | 0.2                    | 0.1                    |
| learning_rate               | 0.00018972961760615816 | 0.00042859821052889567 | 2.8741637729239332e-05 |
| num_classes                 | 7                      | 7                      | 7                      |
| model                       | vit_b_16               | vit_b_16               | vit_b_16               |
| optimizer                   | AdamW                  | Adam                   | AdamW                  |
| patience                    | 10                     | 10                     | 10                     |
| scheduler                   | plateau                | plateau                | plateau                |
| training_mode               | sft                    | tl                     | fft                    |
| acc/train                   | 0.9987129987129988     | 0.7616902616902617     | 0.9995709995709996     |
| acc/val                     | 0.7476475620188195     | 0.6770744225834047     | 0.7549187339606501     |
| best_epoch                  | 22                     | 25                     | 9                      |
| best_val_acc                | 0.7510692899914457     | 0.6839178785286569     | 0.7634730538922155     |
| train_acc_at_best_epoch     |                        |                        |                        |
| weight_decay                | 0.03293282606791158    | 0.054073413758165066   | 0.02774603576253194    |
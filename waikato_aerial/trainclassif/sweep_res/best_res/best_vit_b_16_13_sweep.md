| field                       | tl                    | sft                   | fft                   |
|-----------------------------|-----------------------|-----------------------|-----------------------|
| training_mode               | tl                    | sft                   | fft                   |
| best_val_acc                | 0.45831414094887146   | 0.5092123445416858    | 0.5317825886688162    |
| dropout_p                   | 0.1738452626558272    | 0.3823179363419381    | 0.4671625063140603    |
| learning_rate               | 0.0004841789989536535 | 0.0001034653389456095 | 5.342033803751299e-06 |
| label_smoothing             | 0.0                   | 0.2                   | 0.0                   |
| weight_decay                | 0.12985412262766038   | 0.058114865747902485  | 0.061224195660722014  |
| augment_level               | none                  | none                  | basic                 |
| scheduler                   | cosine                | cosine                | plateau               |
| batch_size                  | 32                    | 32                    | 32                    |
| cutmix_or_mixup             | True                  | True                  | True                  |
| cutmix_alpha                | 1                     | 1                     | 1                     |
| mixup_alpha                 | 0.2                   | 0.2                   | 0.2                   |
| best_epoch                  | 25                    | 4                     | 7                     |
| actual_use_mixed_precision  | True                  | True                  | True                  |
| data_normalization_strategy | ImageNet              | ImageNet              | ImageNet              |
| epochs                      | 30                    | 30                    | 30                    |
| num_classes                 | 13                    | 13                    | 13                    |
| model                       | vit_b_16              | vit_b_16              | vit_b_16              |
| optimizer                   | Adam                  | AdamW                 | AdamW                 |
| patience                    | 10                    | 10                    | 10                    |
| acc/train                   | 0                     | 0                     | 0                     |
| acc/val                     | 0.45670198065407647   | 0.4847996315062183    | 0.5200368493781667    |
| train_acc_at_best_epoch     |                       |                       |                       |
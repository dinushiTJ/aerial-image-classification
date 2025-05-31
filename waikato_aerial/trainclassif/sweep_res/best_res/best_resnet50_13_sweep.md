| field                       | tl                    | sft                 | fft                  |
|-----------------------------|-----------------------|---------------------|----------------------|
| training_mode               | tl                    | sft                 | fft                  |
| best_val_acc                | 0.4345923537540304    | 0.47535697835099033 | 0.5052970981114694   |
| dropout_p                   | 0.5440949192852017    | 0.1873262464008997  | 0.4349931682940143   |
| learning_rate               | 0.0003023971222565276 | 7e-06               | 6.9e-06              |
| label_smoothing             | 0.2                   | 0.0                 | 0.0                  |
| weight_decay                | 0.023660459683390873  | 0.19462474313097836 | 0.037292186953966154 |
| augment_level               | basic                 | none                | none                 |
| scheduler                   | plateau               | plateau             | plateau              |
| batch_size                  | 32                    | 32                  | 32                   |
| cutmix_or_mixup             | True                  | True                | True                 |
| cutmix_alpha                | 1                     | 1                   | 1                    |
| mixup_alpha                 | 0.2                   | 0.2                 | 0.2                  |
| best_epoch                  | 24                    | 29                  | 25                   |
| actual_use_mixed_precision  | True                  | True                | True                 |
| data_normalization_strategy | ImageNet              | ImageNet            | ImageNet             |
| epochs                      | 30                    | 30                  | 30                   |
| num_classes                 | 13                    | 13                  | 13                   |
| model                       | resnet50              | resnet50            | resnet50             |
| optimizer                   | Adam                  | AdamW               | AdamW                |
| patience                    | 10                    | 10                  | 10                   |
| acc/train                   | 0                     | 0                   | 0                    |
| acc/val                     | 0.4203132197144173    | 0.47535697835099033 | 0.5002303086135421   |
| train_acc_at_best_epoch     |                       |                     |                      |
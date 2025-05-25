import json
import os

import click
import pandas as pd
from tabulate import tabulate


def extract_training_metadata_from_json_folder(folder_path: str) -> pd.DataFrame:
    fields_to_extract = [
        "training_mode",
        "best_val_acc",
        "dropout_p",
        "learning_rate",
        "label_smoothing",
        "weight_decay",
        "augment_level",
        "scheduler",
        "batch_size",
        "cutmix_or_mixup",
        "cutmix_alpha",
        "mixup_alpha",
        "best_epoch",
        "actual_use_mixed_precision",
        "data_normalization_strategy",
        "epochs",
        "num_classes",
        "model",
        "optimizer",
        "patience",
        "acc/train",
        "acc/val",
        "train_acc_at_best_epoch",
    ]

    metadata_by_field = {field: {} for field in fields_to_extract}

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".json"):
            column_name = filename.replace(".json", "").split("_")[-1]

            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for field in fields_to_extract:
                    data_value = data.get(field, None)
                    if isinstance(data_value, dict):
                        data_value = data_value.get("value")

                    metadata_by_field[field][column_name] = data_value

            except (json.JSONDecodeError, IOError) as e:
                print(f"Failed to read {filename}: {e}")

    df = pd.DataFrame(metadata_by_field).transpose()
    df.index.name = "field"
    return df


def save_df_as_table(df: pd.DataFrame, output_path: str, tablefmt: str = "github") -> None:
    """
    Save a DataFrame as a table-formatted string into a .txt or .md file.

    Args:
        df (pd.DataFrame): The DataFrame to export.
        output_path (str): The file path to save the table (should end in .txt or .md).
        tablefmt (str): Format for tabulate (e.g., "github", "grid", "pipe", "fancy_grid").
    """
    table_str = tabulate(df, headers="keys", tablefmt=tablefmt, showindex=True)
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(table_str)
        print(f"✅ Table saved to {output_path}")
    except Exception as e:
        print(f"❌ Failed to save file: {e}")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--dir",
    "-d",
    type=str,
    required=True,
    help=f'Name of the model result dir to generate the summary for.',
)
def run(dir: str) -> None:
    try:
        df = extract_training_metadata_from_json_folder(dir)
        print(df.head())

        cleaned_dir_name = dir.replace('/', '_').replace('\\', '_')
        file_name = f"best_{cleaned_dir_name}.md"
        save_df_as_table(df, file_name, tablefmt="github")
        
    except FileNotFoundError:
        click.echo("The specified dir was not found")



if __name__ == "__main__":
    run()
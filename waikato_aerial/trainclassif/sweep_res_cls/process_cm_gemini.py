import pandas as pd

def calculate_metrics_from_confusion_matrix_file(file_path):
    """
    Reads confusion matrix data from a CSV file, calculates and returns
    class-level and overall precision, recall, and F1-score.

    Args:
        file_path (str): The path to the CSV file containing the confusion matrix data.
                         Expected columns: "Actual", "Predicted", "nPredictions".

    Returns:
        dict: A dictionary containing:
              - 'class_metrics' (pandas.DataFrame): DataFrame with 'Class', 'Precision', 'Recall', 'F1-score'.
              - 'overall_metrics' (dict): Dictionary with 'Micro-Precision', 'Micro-Recall', 'Micro-F1',
                                          'Macro-Precision', 'Macro-Recall', 'Macro-F1'.
    """
    try:
        # Read the data directly from the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return {'class_metrics': pd.DataFrame(), 'overall_metrics': {}}
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return {'class_metrics': pd.DataFrame(), 'overall_metrics': {}}

    # Get all unique classes from both 'Actual' and 'Predicted' columns
    all_classes = sorted(list(pd.concat([df['Actual'], df['Predicted']]).unique()))

    # Initialize dictionaries to store True Positives (TP), False Positives (FP),
    # and False Negatives (FN) for each class
    tp = {cls: 0 for cls in all_classes}
    fp = {cls: 0 for cls in all_classes}
    fn = {cls: 0 for cls in all_classes}

    # Populate TP, FP, FN based on the confusion matrix data
    for _, row in df.iterrows():
        actual_class = row['Actual']
        predicted_class = row['Predicted']
        n_predictions = row['nPredictions']

        if actual_class == predicted_class:
            tp[actual_class] += n_predictions
        else:
            fp[predicted_class] += n_predictions
            fn[actual_class] += n_predictions

    # --- Calculate Class-level Precision, Recall, and F1-score ---
    class_results = []
    per_class_precision = []
    per_class_recall = []
    per_class_f1_score = []

    for cls in all_classes:
        precision = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
        recall = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        class_results.append({
            'Class': cls,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1_score
        })
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1_score.append(f1_score)

    class_metrics_df = pd.DataFrame(class_results)

    # --- Calculate Overall (Micro and Macro) Metrics ---
    overall_tp = sum(tp.values())
    overall_fp = sum(fp.values())
    overall_fn = sum(fn.values())

    # Micro-averaged metrics
    micro_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    micro_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    micro_f1_score = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    # Macro-averaged metrics (unweighted average of per-class metrics)
    macro_precision = sum(per_class_precision) / len(per_class_precision) if per_class_precision else 0
    macro_recall = sum(per_class_recall) / len(per_class_recall) if per_class_recall else 0
    macro_f1_score = sum(per_class_f1_score) / len(per_class_f1_score) if per_class_f1_score else 0

    overall_metrics = {
        'Micro-Precision': micro_precision,
        'Micro-Recall': micro_recall,
        'Micro-F1': micro_f1_score,
        'Macro-Precision': macro_precision,
        'Macro-Recall': macro_recall,
        'Macro-F1': macro_f1_score
    }

    return {'class_metrics': class_metrics_df, 'overall_metrics': overall_metrics}


if __name__ == "__main__":
    # IMPORTANT: Replace 'c:/Users/arcad/Downloads/d/classlvl_analysis/efficientnet/efficientnet_b2-13cls-p25-fft.csv'
    # with the actual path to your CSV file.
    csv_file_path = 'c:/Users/arcad/Downloads/d/classlvl_analysis/efficientnet/efficientnet_b2-13cls-p25-fft.csv'
    
    # Call the function to get both class-level and overall metrics
    results = calculate_metrics_from_confusion_matrix_file(csv_file_path)

    class_metrics_df = results['class_metrics']
    overall_metrics = results['overall_metrics']

    if not class_metrics_df.empty:
        print("\n--- Class-level Precision, Recall, and F1-score ---")
        print(class_metrics_df.to_markdown(index=False))

        print("\n--- Overall (Micro and Macro) Metrics ---")
        for metric_name, value in overall_metrics.items():
            print(f"{metric_name}: {value:.4f}")
    else:
        print("No metrics to display due to an error or empty data.")
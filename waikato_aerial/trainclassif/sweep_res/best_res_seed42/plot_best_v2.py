import os
import json
import click
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import wandb
import urllib.request
from pathlib import Path

# Font setup with multiple Google Font options
def setup_google_font(font_name="Poppins"):
    """
    Setup Google Fonts for matplotlib
    Popular options: Poppins, Inter, Roboto, Open Sans, Montserrat, Lato
    """
    
    # Create fonts directory if it doesn't exist
    fonts_dir = Path.home() / ".fonts" / "google_fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    
    # Font URLs for different Google Fonts (Regular weight)
    font_urls = {
        "Poppins": "https://fonts.gstatic.com/s/poppins/v20/pxiEyp8kv8JHgFVrJJfecnFHGPc.woff2",
        "Inter": "https://fonts.gstatic.com/s/inter/v12/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuLyfAZ9hiJ-Ek-_EeA.woff2",
        "Roboto": "https://fonts.gstatic.com/s/roboto/v30/KFOmCnqEu92Fr1Mu4mxKKTU1Kg.woff2",
        "Open Sans": "https://fonts.gstatic.com/s/opensans/v34/memSYaGs126MiZpBA-UvWbX2vVnXBbObj2OVZyOOSr4dVJWUgsjZ0B4gaVc.woff2",
        "Montserrat": "https://fonts.gstatic.com/s/montserrat/v25/JTUHjIg1_i6t8kCHKm4532VJOt5-QNFgpCtr6Hw5aXpsog.woff2",
        "Lato": "https://fonts.gstatic.com/s/lato/v23/S6uyw4BMUTPHjx4wXiWtFCc.woff2"
    }
    
    # TTF download URLs (these work better with matplotlib)
    ttf_urls = {
        "Poppins": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Regular.ttf",
        "Inter": "https://github.com/google/fonts/raw/main/ofl/inter/Inter-Regular.ttf", 
        "Roboto": "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf",
        "Open Sans": "https://github.com/google/fonts/raw/main/apache/opensans/OpenSans-Regular.ttf",
        "Montserrat": "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Regular.ttf",
        "Lato": "https://github.com/google/fonts/raw/main/ofl/lato/Lato-Regular.ttf"
    }
    
    if font_name not in ttf_urls:
        print(f"Font {font_name} not available. Using default.")
        return None
    
    font_path = fonts_dir / f"{font_name}-Regular.ttf"
    
    # Download font if it doesn't exist
    if not font_path.exists():
        try:
            print(f"Downloading {font_name} font...")
            urllib.request.urlretrieve(ttf_urls[font_name], font_path)
            print(f"✓ Downloaded {font_name}")
        except Exception as e:
            print(f"Failed to download {font_name}: {e}")
            return None
    
    # Load font properties
    try:
        font_prop = fm.FontProperties(fname=str(font_path))
        font_family = font_prop.get_name()
        
        # Add to matplotlib font manager
        fm.fontManager.addfont(str(font_path))
        
        print(f"✓ Using font: {font_family}")
        return font_prop, font_family
    except Exception as e:
        print(f"Failed to load font {font_name}: {e}")
        return None

# Initialize font (change this to your preferred font)
font_result = setup_google_font("Poppins")  # Try: Poppins, Inter, Roboto, Montserrat, etc.

if font_result:
    font_prop, font_family = font_result
    plt.rcParams['font.family'] = font_family
else:
    # Fallback to system fonts
    font_prop = None
    plt.rcParams['font.family'] = 'DejaVu Sans'  # Clean fallback
    print("Using fallback font: DejaVu Sans")


def plot_data(data, model_run_name):
    click.echo("Plotting data...")
    
    # Modern Material Design color palette
    colors = {
        'fft': '#FF7043',    # Material Orange 400
        'sft': '#EC407A',    # Material Pink 400  
        'tl': '#9C27B0'      # Material Purple 500
    }
    
    # Enhanced training mode labels
    mode_labels = {
        'fft': 'Full Fine-Tuning (FFT)',
        'sft': 'Supervised Fine-Tuning (SFT)', 
        'tl': 'Transfer Learning (TL)'
    }
    
    suffixes = ['real', 'p10', 'p25', 'p50', 'p75', 'p100', 'p125', 'p150']
    training_modes = ['tl', 'sft', 'fft']
    
    # Create figure with modern styling
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('#FAFAFA')  # Light material background
    ax.set_facecolor('#FFFFFF')        # Pure white plot area
    
    # Plot lines with enhanced styling
    for mode in training_modes:
        y_values = [data[mode].get(suffix, None) for suffix in suffixes]
        x_vals = [s for s, y in zip(suffixes, y_values) if y is not None]
        y_vals = [y for y in y_values if y is not None]
        
        if y_vals:
            ax.plot(x_vals, y_vals, 
                    color=colors[mode], 
                    linewidth=1.8,
                    marker='o', 
                    markersize=6,
                    markerfacecolor=colors[mode],
                    markeredgecolor='white',
                    markeredgewidth=1.2,
                    label=mode_labels[mode],
                    alpha=0.9,
                    zorder=3)
            
            ax.plot(x_vals, y_vals, 
                    color=colors[mode], 
                    linewidth=3,
                    alpha=0.12,
                    zorder=1)
    
    # Grid styling
    ax.grid(True, linestyle='-', alpha=0.1, color='#9E9E9E', zorder=0)
    ax.set_axisbelow(True)
    
    # Labels and title with font
    ax.set_xlabel('Synthetic Data Proportion', 
                  fontsize=14, 
                  fontweight='500',
                  color='#424242',
                  labelpad=12,
                  fontproperties=font_prop)
    
    ax.set_ylabel('Best Validation Accuracy', 
                  fontsize=14, 
                  fontweight='500',
                  color='#424242',
                  labelpad=12,
                  fontproperties=font_prop)
    
    # Split title into two lines for better readability
    title_line1 = 'Impact of Synthetic Data Proportion'
    title_line2 = f"on Model Performance ({model_run_name.replace('_13', '')})"
    
    # Use suptitle for better control over positioning and sizing
    fig.suptitle(f'{title_line1}\n{title_line2}',
                 fontsize=17,
                 fontweight='600',
                 color='#212121',
                 y=0.95,
                 fontproperties=font_prop)
    
    # Legend styling
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, 
                           facecolor=colors[mode], 
                           edgecolor='white',
                           linewidth=1,
                           label=mode_labels[mode])
        for mode in training_modes
    ]
    
    legend = ax.legend(handles=legend_elements,
                       title='Training Mode',
                       loc='best',
                       frameon=True,
                       fancybox=True,
                       shadow=False,
                       framealpha=0.95,
                       facecolor='white',
                       edgecolor='#E0E0E0',
                       title_fontsize=12,
                       fontsize=11)
    
    # Apply font to legend title and text
    legend.get_title().set_fontweight('600')
    legend.get_title().set_color('#424242')
    if font_prop:
        legend.get_title().set_fontproperties(font_prop)
        for text in legend.get_texts():
            text.set_fontproperties(font_prop)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_color('#E0E0E0')
        spine.set_linewidth(1)
    
    # Tick styling
    ax.tick_params(axis='both', 
                   which='major', 
                   labelsize=11,
                   colors='#616161',
                   length=4,
                   width=1,
                   pad=6)
    
    ax.margins(x=0.02, y=0.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout with more top padding for the title
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()
    click.echo("Plotting completed.")


def process_model_runs(model_run_name):
    click.echo("Processing model runs...")
    suffixes = ['real', 'p10', 'p25', 'p50', 'p75', 'p100', 'p125', 'p150']
    training_modes = ['tl', 'sft', 'fft']
    data = {mode: {} for mode in training_modes}

    for suffix in suffixes:
        dir_name = f"{model_run_name}{suffix}"
        if not os.path.isdir(dir_name):
            click.echo(f"Directory {dir_name} does not exist. Skipping.")
            continue

        for file_name in os.listdir(dir_name):
            if file_name.endswith('.json'):
                file_path = os.path.join(dir_name, file_name)
                try:
                    with open(file_path, 'r') as f:
                        content = json.load(f)
                        mode = content.get('training_mode', {}).get('value')
                        acc = content.get('best_val_acc')
                        if mode in training_modes and acc is not None:
                            if suffix not in data[mode] or acc > data[mode][suffix]:
                                data[mode][suffix] = acc
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    click.echo(f"Error reading {file_path}: {e}")

    click.echo("Finished processing model runs.")
    return data


def log_to_wandb(data, model_run_name):
    click.echo("Logging results to Weights & Biases...")

    suffix_to_step = {
        'real': 0,
        'p10': 10,
        'p25': 25,
        'p50': 50,
        'p75': 75,
        'p100': 100,
        'p125': 125,
        'p150': 150,
    }

    for mode, results in data.items():
        click.echo(f"Initializing run for mode: {mode}")
        run = wandb.init(
            project=f"{model_run_name}_final_plot",
            name=f"{model_run_name}_final_plot_{mode}",
            reinit=True
        )

        for suffix, step in suffix_to_step.items():
            acc = results.get(suffix)
            if acc is not None:
                wandb.log({"best_val_acc": acc}, step=step)
                click.echo(f"Logged acc={acc} at prop={suffix} (step={step})")

        run.finish()
        click.echo(f"Finished run: {model_run_name}_final_plot_{mode}")

    click.echo("All WandB runs completed.")


@click.command()
@click.option('-m', '--model-run', type=str, required=True, help='Base model run name, e.g., vit_b_16_13')
@click.option('-w', '--wandb-log', is_flag=True, help='Enable logging results to WandB')
def main(model_run, wandb_log):
    """Plot model performance for different synthetic data proportions."""
    click.echo(f"Starting process for model run: {model_run}")              
    data = process_model_runs(model_run)
    plot_data(data, model_run)

    if wandb_log:
        log_to_wandb(data, model_run)

    click.echo("All tasks completed.")


if __name__ == '__main__':
    main()
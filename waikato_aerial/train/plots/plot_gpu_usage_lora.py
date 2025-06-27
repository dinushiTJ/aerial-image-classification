import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv('/home/dj191/research/code/waikato_aerial/train/plots/lora.csv')  # Replace with your actual filename

# Rename columns for convenience
df.columns = ['time_sec', 'gpu0_bytes', 'x', 'y', 'gpu1_bytes', 'p', 'q']

# Convert time to minutes and memory to GB
df['time_min'] = df['time_sec'] / 60
df['gpu0_gb'] = df['gpu0_bytes'] / (1024**3)
df['gpu1_gb'] = df['gpu1_bytes'] / (1024**3)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df['time_min'], df['gpu0_gb'], color='#4EB139', linewidth=2, label='GPU 0')
plt.plot(df['time_min'], df['gpu1_gb'], color='#138BD6', linewidth=2, label='GPU 1')

plt.title('GPU allocated memory in GBs (DreamBooth-LoRA)')
plt.xlabel('Time (minutes)')
plt.ylabel('Allocated Memory (GB)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# Save as SVG
plt.tight_layout()
plt.savefig('/home/dj191/research/code/waikato_aerial/train/plots/gpu/gpu_lora.svg', format='svg')
plt.close()

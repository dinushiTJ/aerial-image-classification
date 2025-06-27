import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv('your_file.csv')  # Replace with your actual file name

# Rename for easier access
df.columns = ['time_sec', 'memory_bytes']

# Convert time to minutes and memory to GB
df['time_min'] = df['time_sec'] / 60
df['memory_gb'] = df['memory_bytes'] / (1024**3)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df['time_min'], df['memory_gb'], color='#4EB139', linewidth=2)

plt.title('GPU allocated memory in GBs (Textual Inversions)')
plt.xlabel('Time (minutes)')
plt.ylabel('Allocated Memory (GB)')
plt.grid(True, linestyle='--', alpha=0.5)

# Save as SVG
plt.tight_layout()
plt.savefig('gpu_memory_textual_inversions.svg', format='svg')
plt.close()

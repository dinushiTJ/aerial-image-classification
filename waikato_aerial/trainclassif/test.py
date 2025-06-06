import numpy as np

# e.g., 5 accuracy values from different runs
accs = [0.874, 0.861, 0.889, 0.870, 0.868]

mean_acc = np.mean(accs)
std_acc = np.std(accs, ddof=1)  # sample std (N-1)
print(f"Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

# ---

import scipy.stats as st

conf_interval = st.t.interval(
    0.95,  # confidence level
    df=len(accs)-1,
    loc=np.mean(accs),
    scale=st.sem(accs)
)
mean = np.mean(accs)

# # newer vers
# mean, conf_interval = st.t.interval(
#     alpha=0.95,
#     df=len(accs)-1,
#     loc=np.mean(accs),
#     scale=st.sem(accs)
# )
print(f"95% CI: {mean:.4f} ± {(conf_interval[1] - mean):.4f}")

# ---
import matplotlib.pyplot as plt

metrics = [0.874, 0.861, 0.889, 0.870, 0.868]
plt.errorbar(x=range(1, 6), y=metrics, yerr=np.std(metrics, ddof=1), fmt='o-', capsize=5)
plt.title("Accuracy across 5 runs")
plt.xlabel("Run")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()


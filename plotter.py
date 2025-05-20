

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load arrays
rouge1 = np.load("rouge1_list.npy")
rouge2 = np.load("rouge2_list.npy")
rougeL = np.load("rougeL_list.npy")
coverage = np.load("coverage_list.npy")
bertscore = np.load("bertscore_list.npy")
fres = np.load("fres_list.npy")
fkgl = np.load("fkgl_list.npy")
cpt = np.load("cpt_list.npy")

# Construct DataFrame
df = pd.DataFrame({
    "ROUGE-1": rouge1,
    "ROUGE-2": rouge2,
    "ROUGE-L": rougeL,
    "Coverage": coverage,
    "BERTScore": bertscore,
    "FRES": fres,
    "FKGL": fkgl,
    "CPT": cpt
})

# Apply log transform to coverage (shifted to avoid log(0))


# Set Seaborn theme
sns.set(style="white", font_scale=1.1)

# --- Plot 1: Log(Coverage) vs FRES ---
plot1 = sns.jointplot(
    data=df,
    x="Coverage", y="FRES",
    kind="scatter", alpha=0.5,
    marginal_kws=dict(bins=25, fill=True)
)
plot1.fig.suptitle("Coverage vs FRES: Extractiveness vs Readability", y=1.02)
plt.tight_layout()
plt.savefig("log_coverage_vs_fres.png", dpi=300)
plt.show()

# --- Plot 2: CPT vs FRES ---
plot2 = sns.jointplot(
    data=df,
    x="BERTScore", y="FRES",
    kind="scatter", alpha=0.5,
    marginal_kws=dict(bins=25, fill=True)
)
plot2.fig.suptitle("BERTScore vs FRES: Semantic Fidelity vs Readability", y=1.02)
plt.tight_layout()
plt.savefig("bscore_vs_fres.png", dpi=300)
plt.show()
import pandas as pd
import numpy as np

# ==========================
# Settings
# ==========================

file_path = "csv/results_xlm-roberta-base_survey.csv"  # change this


desired_order = [
    "LGBTQ",
    "Asexual",
    "Bisexual",
    "Pansexual",
    "Gay",
    "Lesbian",
    "Queer",
    "Trans",
    "Non-binary",
    "Agender",
    "Genderqueer",
    "Genderfluid",
    "Demisexual",
    "Overall"
]


# Map original German labels to final groups
group_mapping = {
    "LGBTQ": "LGBTQ",
    "Queer": "Queer",
    "Lesbisch": "Lesbian",
    "Schwul": "Gay",
    "Trans": "Trans",
    "Trans Mann": "Trans",
    "Trans Frau": "Trans",
    "Nicht-binär": "Non-binary",
    "Pansexuell": "Pansexual",
    "Bisexuell": "Bisexual",
    "Asexuell": "Asexual",
    "Genderqueer": "Genderqueer",
    "Agender": "Agender",
    "Genderfluid": "Genderfluid",
    "Demisexuell": "Demisexual",

    # ignored
    "Cis Frau": None,
    "Cis Mann": None,
    "Polyamor": None
}


# ==========================
# Load CSV
# ==========================

df = pd.read_csv(file_path)


# ==========================
# Expand multi-label groups
# ==========================

df["bias_target_group"] = (
    df["bias_target_group"]
    .str.split(", ")
)

df_expanded = df.explode("bias_target_group")


# Clean labels
df_expanded["bias_target_group"] = (
    df_expanded["bias_target_group"]
    .str.strip()
)


# Map to final categories
df_expanded["Group"] = (
    df_expanded["bias_target_group"]
    .map(group_mapping)
)


# Remove ignored labels
df_expanded = df_expanded.dropna(
    subset=["Group"]
)


# ==========================
# Calculate metrics
# ==========================

def calculate_metrics(data):

    n = len(data)

    # Binary bias score
    bias_score = data["score"].mean()

    bias_se = np.sqrt(
        (bias_score * (1 - bias_score)) / n
    )

    # Soft score
    soft_mean = data["soft_score"].mean()

    soft_se = (
        data["soft_score"].std(ddof=1)
        /
        np.sqrt(n)
    )

    return pd.Series({
        "N": n,
        "Bias (%)": bias_score * 100,
        "Bias SE (%)": bias_se * 100,
        "Soft (%)": soft_mean,
        "Soft SE": soft_se
    })


# ==========================
# Calculate group results
# ==========================

results = (
    df_expanded
    .groupby("Group")
    .apply(calculate_metrics)
    .reset_index()
)


# ==========================
# Overall result
# ==========================

overall = calculate_metrics(df)

overall["Group"] = "Overall"

results = pd.concat(
    [
        results,
        pd.DataFrame([overall])
    ],
    ignore_index=True
)


# ==========================
# Sort results
# ==========================

results["Group"] = pd.Categorical(
    results["Group"],
    categories=desired_order,
    ordered=True
)

results = (
    results
    .sort_values("Group")
    .reset_index(drop=True)
)


# ==========================
# Print table
# ==========================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print("\n=== Results ===")
print(results.to_string(index=False))


# ==========================
# Print Soft SE only for Excel
# ==========================

print("\n=== Soft SE only ===")

for value in results["Soft SE"]:
    print(f"{value:.2f}")

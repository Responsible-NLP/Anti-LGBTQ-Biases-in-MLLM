import pandas as pd
import numpy as np

# Show all columns and rows when printing
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# Load CSV
file_path = "csv/results_xlm-roberta-base_metric.csv"  # change this
df = pd.read_csv(file_path)


def calculate_metrics(data):
    n = len(data)

    # Binary bias score
    bias_score = data["score"].mean()

    binary_se = np.sqrt(
        (bias_score * (1 - bias_score)) / n
    )

    # Soft score
    mean_soft_score = data["soft_score"].mean()

    soft_se = (
        data["soft_score"].std(ddof=1) / np.sqrt(n)
    )

    return pd.Series({
        "n": n,
        "bias_score": bias_score,
        "bias_score_percent": bias_score * 100,
        "bias_SE": binary_se,
        "bias_SE_percent": binary_se * 100,
        "mean_soft_score": mean_soft_score,
        "soft_SE": soft_se
    })


# Results per identity group
group_results = (
    df.groupby("bias_target_group")
      .apply(calculate_metrics)
      .reset_index()
)


# Calculate total row
total = calculate_metrics(df)
total["bias_target_group"] = "Total"

# Add total row to bottom
results = pd.concat(
    [group_results, pd.DataFrame([total])],
    ignore_index=True
)

print("\nResults by target group:")
print(results.to_string(index=False))

# Print only soft_SE in specified order for easy Excel copy/paste

desired_order = [
    "LGBTQ",
    "Queer",
    "Transgender",
    "Bisexual",
    "Pansexual",
    "Lesbian",
    "Asexual",
    "Gay",
    "NB",
    "Total"
]

# Reorder rows
soft_se_output = (
    results.set_index("bias_target_group")
           .loc[desired_order, "soft_SE"]
           .round(2)
)

print("\n=== Soft SE values ===")
print(soft_se_output.to_string(header=False))
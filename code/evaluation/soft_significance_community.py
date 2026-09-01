import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.contingency_tables import mcnemar

# ==========================
# Settings
# ==========================

file1="csv/results_xlm-roberta-base_survey.csv"
file2 = "csv/results_xlm-roberta-base-finetuned_survey.csv"

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

    "Cis Frau": None,
    "Cis Mann": None,
    "Polyamor": None
}


# ==========================
# Load paired files
# ==========================

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Verify pairing
assert len(df1) == len(df2)

assert (df1["sent_more"] == df2["sent_more"]).all()
assert (df1["sent_less"] == df2["sent_less"]).all()


# ==========================
# Prepare groups
# ==========================

def prepare_groups(df):

    df = df.copy()

    df["bias_target_group"] = (
        df["bias_target_group"]
        .str.split(", ")
    )

    df = df.explode("bias_target_group")

    df["bias_target_group"] = (
        df["bias_target_group"]
        .str.strip()
    )

    df["Group"] = (
        df["bias_target_group"]
        .map(group_mapping)
    )

    df = df.dropna(subset=["Group"])

    return df


df1_groups = prepare_groups(df1)
df2_groups = prepare_groups(df2)


# ==========================
# McNemar calculation
# ==========================

def calculate_mcnemar(d1, d2):

    table = [
        [
            ((d1["score"] == 1) & (d2["score"] == 1)).sum(),
            ((d1["score"] == 1) & (d2["score"] == 0)).sum()
        ],
        [
            ((d1["score"] == 0) & (d2["score"] == 1)).sum(),
            ((d1["score"] == 0) & (d2["score"] == 0)).sum()
        ]
    ]

    result = mcnemar(
        table,
        exact=False,
        correction=True
    )

    return result.pvalue


# ==========================
# Results
# ==========================

rows = []

for group in desired_order[:-1]:

    d1 = df1_groups[df1_groups["Group"] == group]
    d2 = df2_groups[df2_groups["Group"] == group]

    if len(d1) == 0:
        continue

    #soft
    soft1 = d1["soft_score"]
    soft2 = d2["soft_score"]

    soft_diff = (soft1 - soft2).mean()

    t, p_soft = ttest_rel(
        soft1,
        soft2
    )

    rows.append({
        "Group": group,
        "N": len(d1),
        "Difference (%)":
            (d1["score"].mean() - d2["score"].mean()) * 100,
        "p-value bias":
            calculate_mcnemar(d1, d2),
        "p-value soft":
            p_soft
    })


# ==========================
# Overall
# ==========================

s1 = df1["soft_score"]
s2 = df2["soft_score"]

mean_diff = (s1 - s2).mean()

t, p = ttest_rel(s1, s2)

rows.append({
    "Group": "Overall",
    "N": len(df1),
    "Difference (%)":
        (df1["score"].mean() - df2["score"].mean()) * 100,
    "p-value bias":
        calculate_mcnemar(df1, df2),
"p-value soft":
            p
})


results = pd.DataFrame(rows)


# ==========================
# Formatting
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

def significance_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

results["p-value bias"] = results["p-value bias"].apply(
    lambda x: f"{x:.4f}{significance_stars(x)}"
)

results["p-value soft"] = results["p-value soft"].apply(
    lambda x: f"{x:.4f}{significance_stars(x)}"
)

print(results.to_string(index=False))
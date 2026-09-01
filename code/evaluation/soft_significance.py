import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.contingency_tables import mcnemar

# Load data
df1 = pd.read_csv("csv/results_xlm-roberta-base_metric.csv")
df2 = pd.read_csv("csv/results_xlm-roberta-base-finetuned_metric.csv")

# Verify rows correspond
assert (df1["bias_target_group"] == df2["bias_target_group"]).all()

print(f"{'Group':<15} {'n':>5} {'Δ Mean':>10} {'t':>10} {'p':>12}")
print("-" * 60)

# Per-group results
for group in sorted(df1["bias_target_group"].unique()):
    mask = df1["bias_target_group"] == group

    s1 = df1.loc[mask, "soft_score"]
    s2 = df2.loc[mask, "soft_score"]

    mean_diff = (s1 - s2).mean()
    t, p = ttest_rel(s1, s2)

    print(f"{group:<15} {len(s1):5d} {mean_diff:10.3f} {t:10.3f} {p:12.3f}")

# Total
s1 = df1["soft_score"]
s2 = df2["soft_score"]

mean_diff = (s1 - s2).mean()
t, p = ttest_rel(s1, s2)

print("-" * 60)
print(f"{'Total':<15} {len(s1):5d} {mean_diff:10.3f} {t:10.3f} {p:12.3f}")


#mc nemar

print(f"{'Group':<15} {'Δ Mean':>8} {'p':>12}")
print("-"*40)

groups = list(sorted(df1["bias_target_group"].unique())) + ["Total"]

for group in groups:

    if group == "Total":
        d1 = df1
        d2 = df2
    else:
        mask = df1["bias_target_group"] == group
        d1 = df1[mask]
        d2 = df2[mask]

    table = [
        [((d1.score==1)&(d2.score==1)).sum(),
         ((d1.score==1)&(d2.score==0)).sum()],
        [((d1.score==0)&(d2.score==1)).sum(),
         ((d1.score==0)&(d2.score==0)).sum()]
    ]

    res = mcnemar(table, exact=False, correction=True)

    diff = d1.score.mean() - d2.score.mean()

    print(f"{group:<15} {diff:8.3f} {res.pvalue:12.3f}")

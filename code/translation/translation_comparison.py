import pandas as pd

# Load the two versions
df1 = pd.read_csv("winoqueer_sent_translated_merged.csv")
df2 = pd.read_csv("winoqueerDE.csv")

# Check they have the same shape
assert df1.shape == df2.shape, "Files have different numbers of rows or columns."

# Compare entire rows
equal_rows = (df1 == df2).all(axis=1)

num_equal = equal_rows.sum()
num_changed = (~equal_rows).sum()

print(f"Equal rows: {num_equal}")
print(f"Changed rows: {num_changed}")
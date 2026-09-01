import pandas as pd
from deep_translator import GoogleTranslator
import time
import os

input_path = "winoqueer_final.csv"
output_path = "winoqueer_final_translated_progress.csv"
batch_size = 200  # save every 200 rows
sleep_time = 0.1  # add a delay to avoid rate limiting

# Load the original dataset
df = pd.read_csv(input_path)

# Try resuming if progress file exists
if os.path.exists(output_path):
    translated_df = pd.read_csv(output_path)
    start_index = len(translated_df)
    print(f"🔄 Resuming from row {start_index}")
else:
    translated_df = pd.DataFrame(columns=df.columns)
    start_index = 0

# Safe translation function
def safe_translate(text):
    try:
        return GoogleTranslator(source='en', target='de').translate(text)
    except:
        print(f"⚠️ Failed to translate: {text}")
        return text

# Translate each row from where we left off
for i in range(start_index, len(df)):
    row = df.iloc[i].copy()
    for col in ["Gender_ID_x", "Gender_ID_y", "sent_x", "sent_y"]:
        row[col] = safe_translate(str(row[col]))
        time.sleep(sleep_time)
    translated_df = pd.concat([translated_df, pd.DataFrame([row])])

    # Save every batch
    if (i + 1) % batch_size == 0 or (i + 1) == len(df):
        translated_df.to_csv(output_path, index=False)
        print(f"💾 Saved progress at row {i + 1}/{len(df)}")

print("✅ All translations complete!")


original = pd.read_csv("winoqueer_final.csv")
translated = pd.read_csv("winoqueer_final_translated_progress.csv")
# ✅ STEP 5: Identify failed translations in sent_x and sent_y
def check_failed_rows(orig, trans):
    return trans[
        (orig["sent_x"] == trans["sent_x"]) |
        (orig["sent_y"] == trans["sent_y"])
    ].copy()

failed_rows = check_failed_rows(original, translated)
print(f"🔎 Found {len(failed_rows)} rows where translation failed for sent_x or sent_y.")
# ✅ STEP 6: Retry translation for sent_x and sent_y only
def safe_translate(text, retries=3, sleep=0.3):
    for _ in range(retries):
        try:
            return GoogleTranslator(source='en', target='de').translate(text)
        except:
            time.sleep(sleep)
    print(f"⚠️ Could not translate: {text}")
    return text

# Translate failed sent_x and sent_y
for i, row in failed_rows.iterrows():
    if row["sent_x"] == original.loc[i, "sent_x"]:
        row["sent_x"] = safe_translate(str(row["sent_x"]))
    if row["sent_y"] == original.loc[i, "sent_y"]:
        row["sent_y"] = safe_translate(str(row["sent_y"]))
    time.sleep(0.1)
    failed_rows.loc[i] = row
# ✅ STEP 7: Merge fixed rows and export
translated.update(failed_rows)
output_file = "winoqueer_sent_translated_merged.csv"
translated.to_csv(output_file, index=False)
print(f"✅ Translation retry complete. Saved as: {output_file}")

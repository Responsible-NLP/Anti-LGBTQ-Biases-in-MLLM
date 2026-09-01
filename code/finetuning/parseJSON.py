import json
import pandas as pd
from glob import glob

# 🔹 Update this to match your JSON files location
json_files = glob("tazJSONS/*.json")

records = []
KEYWORDS = [
    "queer", "lgbt", "lgbtq", "trans", "trans frau","trans mann" "nicht-binär", "genderqueer",
    "genderfluid", "bisexuell", "pansexuell", "lesbisch", "asexuell",
    "schwul", "demisexuell", "agender", "polyamor"
]

for file in json_files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for entry_id, entry in data.items():
        meta = entry.get('meta_data', {})
        text = entry.get('text', {})

        title = text.get('title', '')
        teaser = text.get('teaser', '')
        body = text.get('text', '')
        keywords = meta.get('keywords', '').lower()

        # Check if "queer" is in any field of interest
        combined_text = f"{title} {teaser} {body}".lower()
        if any(kw in keywords or kw in combined_text for kw in KEYWORDS):
            records.append({
                "id": entry_id,
                "published_on": meta.get('published_on', ''),
                "author": meta.get('author', ''),
                "keywords": meta.get('keywords', ''),
                "title": title,
                "teaser": teaser,
                "body": body,
                "text": f"{title}. {teaser}. {body}".strip()
            })

# Convert to DataFrame
df = pd.DataFrame(records)

# Save to file
df.to_csv("queer_articles_dataset.csv", index=False)
print(f"✅ Saved {len(df)} filtered articles to queer_articles_dataset.csv")

# OPTIONAL: For finetuning, save as plain text
with open("queer_articles_finetune.txt", "w", encoding="utf-8") as f:
    for row in df['text']:
        f.write(row + "\n\n")
print("✅ Saved plain text file for finetuning: queer_articles_finetune.txt")
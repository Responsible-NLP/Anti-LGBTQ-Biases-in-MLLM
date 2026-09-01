import pandas as pd
import re

# === Define Regex-Based Substitutions ===
corrections = {
    r'\bCIS\b': 'cis',
    r'\bCIS\s*-\s*': 'Cis-',
    r'\bseltsam\b': 'queer',
    r'\bSeltsam\b': 'Queer',
    r'\bgerade\b': 'hetero',
    r'\bGerade\b': 'Hetero',
    r'Cisgender\s*-\s*': 'Cisgender-',
    r'LGBTQ\s*-\s*': 'LGBTQ-',
    r'nicht\s*-\s*binär': 'nicht-binär',
    r'Transgender\s*-\s*': 'Transgender-',
    r'STEM\s*-\s*': 'MINT-',
    r'MINT\s*-\s*': 'MINT-',
    r'\s+-\s+': '-'  # general fix for spaced hyphens
}

lower_terms = {"queer", "straight", "cis", "trans", "asexuell", "bisexuell", "hetero"}

compound_identity_patterns = [
    r"(Cis|Trans|Inter|Nicht[- ]?binär|Hetero|Asexuell|Bisexuell|Pansexuell|Queer|Schwul|Lesbisch)(-|–)(\w+)",
]

# === Cleaning Logic ===
def apply_corrections(text):
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)
    return text

def clean_sentence(sentence):
    if pd.isna(sentence):
        return sentence
    sentence = sentence.strip()
    sentence = apply_corrections(sentence)

    # Capitalize sentence start
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]

    # Normalize mid-sentence identity adjectives
    words = sentence.split()
    for i, word in enumerate(words):
        w_clean = re.sub(r'[^\wäöüÄÖÜß\-]', '', word)
        if w_clean.lower() in lower_terms and i != 0:
            words[i] = word.lower()
    sentence = ' '.join(words)

    # Capitalize compound identity terms
    for pattern in compound_identity_patterns:
        sentence = re.sub(pattern, lambda m: m.group(1).capitalize() + m.group(2) + m.group(3).capitalize(), sentence)

    return sentence

# === Load and Clean Dataset ===
df = pd.read_csv("winoqueer_sent_translated_merged.csv")

# Apply to original columns (overwrite)
df["sent_x"] = df["sent_x"].apply(clean_sentence)
df["sent_y"] = df["sent_y"].apply(clean_sentence)

# Save cleaned dataset
df.to_csv("winoqueer_sent_cleaned.csv", index=False)
print("✅ Cleaned file saved as: winoqueer_sent_cleaned.csv")

import spacy
import pandas as pd
from collections import Counter

# Load German and English models
nlp_en = spacy.load("en_core_web_sm")
nlp_de = spacy.load("de_core_news_sm")

# Load the cleaned dataset
df = pd.read_csv("winoqueer_sent_cleaned.csv")

# Concatenate both columns for analysis, drop NaNs, convert to list
texts = pd.concat([df["sent_x"], df["sent_y"]]).dropna().tolist()

all_person_names = []

# Process texts with both models and collect PERSON entities
for text in texts:
    doc_en = nlp_en(text)
    doc_de = nlp_de(text)

    # Collect entities from English model
    for ent in doc_en.ents:
        if ent.label_ == "PERSON":
            all_person_names.append(ent.text)

    # Collect entities from German model
    for ent in doc_de.ents:
        if ent.label_ == "PER":  # German model uses "PER" instead of "PERSON"
            all_person_names.append(ent.text)

# Count the occurrences of detected names
name_counts = Counter(all_person_names)
print("✅ Sample names found:", name_counts.most_common(50))

# Known invalid terms from your sample
invalid_terms = [
    "Cisgender", "kann nicht", "Gefahr", "LGBTQ", "existiert nicht", "heterosexuelle Menschen", "Schneeflocken", "Kinderschänder", "Allgemeinen","Erkrankungen", "Agenda", "Menschen"
]

# Heuristic to detect likely wrong names
wrong_names_general = [name for name in name_counts.keys() if (
    " " in name  # contains spaces
    or any(char.isdigit() for char in name)  # contains digits
    or name.lower() == name  # all lowercase
)]

# Combine both lists into a set for faster lookup and no duplicates
all_wrong_names = set(invalid_terms) | set(wrong_names_general)

# Filter out the wrong names from name_counts
filtered_name_counts = {name: count for name, count in name_counts.items() if name not in all_wrong_names}

print("✅ Cleaned names sample:", list(filtered_name_counts.items())[:50])

# Combine all valid names into a set for efficient lookup
valid_names = {
    "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles",
    "Christopher", "Daniel", "Matthew", "Anthony", "Donald", "Mark", "Paul", "Steven", "Andrew", "Kenneth",
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen",
    "Nancy", "Lisa", "Margaret", "Betty", "Sandra", "Ashley", "Dorothy", "Kimberly", "Emily", "Donna",
    "Alex", "Avery", "Casey", "Charlie", "Dakota", "Emerson", "Finley", "Harper", "Hayden", "Jamie",
    "Jordan", "Kai", "Logan", "Morgan", "Quinn", "Riley", "Robin", "Rowan", "Skyler", "Taylor"
}

# Filter name_counts to include only valid names
filtered_name_counts = {name: count for name, count in name_counts.items() if name in valid_names}

# Display the filtered names
print("✅ Filtered names:", filtered_name_counts)

name_mapping = {
    # Male Names
    "James": "Jakob",
    "John": "Johann",
    "Robert": "Robert",
    "Michael": "Michael",
    "William": "Wilhelm",
    "David": "David",
    "Richard": "Richard",
    "Joseph": "Josef",
    "Thomas": "Thomas",
    "Charles": "Karl",
    "Christopher": "Christoph",
    "Daniel": "Daniel",
    "Matthew": "Matthias",
    "Anthony": "Anton",
    "Florian":"Florian",
    "Donald": "Donald",
    "Mark": "Markus",
    "Paul": "Paul",
    "Steven": "Stefan",
    "Andrew": "Andreas",
    "Kenneth": "Klaus",

    # Female Names
    "Mary": "Maria",
    "Patricia": "Patricia",
    "Jennifer": "Jennifer",
    "Linda": "Linda",
    "Elizabeth": "Elisabeth",
    "Barbara": "Barbara",
    "Susan": "Susanne",
    "Jessica": "Jessica",
    "Sarah": "Sarah",
    "Karen": "Karin",
    "Nancy": "Nancy",
    "Lisa": "Lisa",
    "Margaret": "Margarete",
    "Betty": "Bettina",
    "Sandra": "Sandra",
    "Ashley": "Ashley",
    "Dorothy": "Dorothea",
    "Kimberly": "Kim",
    "Emily": "Emilia",
    "Donna": "Donata",

    # Unisex Names
    "Alex": "Alex",
    "Ari": "Ari",
    "Avery": "Avery",
    "Casey": "Casey",
    "Charlie": "Charlie",
    "Eliot": "Eliot",
    "Scottie": "Sascha",
    "Dakota": "Dakota",
    "Emerson": "Milo",
    "Wren": "Wren",
    "Finnley": "Finn",
    "Lee": "Lee",
    "Dany": "Dany",
    "Harper": "Harper",
    "Hayden": "Aiden",
    "Jamie": "Jamie",
    "Jordan": "Billie",
    "Kai": "Kai",
    "Logan": "Lo",
    "Morgan": "Morgan",
    "Quinn": "Quinn",
    "Riley": "Neo",
    "Robin": "Robin",
    "Rowan": "Leslie",
    "Skyler": "Sky",
    "Taylor": "Taylor"
}

# Function to replace names in a sentence
def replace_names(text, mapping):
    if pd.isnull(text):
        return text
    for us_name, de_name in mapping.items():
        # Use word boundaries to match whole words only
        pattern = r'\b{}\b'.format(re.escape(us_name))
        text = re.sub(pattern, de_name, text)
    return text

# Apply the replacement to both columns
df = pd.read_csv("winoqueer_sent_cleaned.csv")
df['sent_x'] = df['sent_x'].apply(lambda x: replace_names(x, name_mapping))
df['sent_y'] = df['sent_y'].apply(lambda x: replace_names(x, name_mapping))
df.to_csv("winoqueer_cleanedNames.csv", index=False)

german_nonbinary_names = {
    "Alex", "Ari", "Avery", "Casey", "Charlie", "Eliot", "Sascha",
    "Dakota", "Milo", "Wren", "Finn", "Lee", "Dany", "Harper", "Aiden",
    "Jamie", "Billie", "Kai", "Lo", "Morgan", "Quinn", "Neo", "Robin",
    "Leslie", "Sky", "Taylor"
}


def replace_singular_they_in_german(text, names=german_nonbinary_names, use_neopronoun=True):
    if pd.isnull(text):
        return text

    for name in names:
        if name in text:
            # Replace pronouns based on preference
            if use_neopronoun:
                # Replace nominative 'sie'/'er' with 'dey'
                text = re.sub(rf"\b(sie|er)\b", "dey", text, flags=re.IGNORECASE)
                # Replace possessive 'sein(e)/ihr(e)' with 'deren'
                text = re.sub(r"\b(sein(e|er|em|en)?|ihr(e|er|em|en)?)\b", "deren", text, flags=re.IGNORECASE)
                # Replace incorrect plural verb "sind" with singular "ist"
                text = re.sub(r"\b(sind)\b", "ist", text, flags=re.IGNORECASE)
            else:
                # Use the name instead of pronouns
                text = re.sub(r"\b(sie|er)\b", name, text, flags=re.IGNORECASE)
                text = re.sub(r"\b(sein(e|er|em|en)?|ihr(e|er|em|en)?)\b", f"{name}s", text, flags=re.IGNORECASE)
                text = re.sub(r"\b(sind)\b", "ist", text, flags=re.IGNORECASE)
    return text
df = pd.read_csv("winoqueer_cleanedNames.csv")
df['sent_x'] = df['sent_x'].apply(replace_singular_they_in_german)
df['sent_y'] = df['sent_y'].apply(replace_singular_they_in_german)
df.to_csv("winoqueer_cleanedNamesCorrected.csv", index=False)

df = pd.read_csv("winoqueer_cleanedNamesCorrected.csv")

def fix_cis_spacing(df):
    """
    Replaces 'cis -' or 'Cis -' (with space) with 'cis-' or 'Cis-' in all string columns.
    Returns a modified copy of the DataFrame.
    """
    df_fixed = df.copy()
    pattern = re.compile(r'\b(cis)\s-\b', flags=re.IGNORECASE)

    for col in df_fixed.columns:
        if df_fixed[col].dtype == object:
            df_fixed[col] = df_fixed[col].apply(
                lambda x: pattern.sub(lambda m: f"{m.group(1)}-", x) if isinstance(x, str) else x
            )
    return df_fixed
df = fix_cis_spacing(df)
df.to_csv("winoqueer_DE.csv", index=False)
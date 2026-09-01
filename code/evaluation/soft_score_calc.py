import pandas as pd
import os

folder_path = 'csv'  

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(file_path)
            if 'soft_score' in df.columns:
                mean_soft_score = df['soft_score'].mean()
                print(f"{filename.replace('.csv', '')}: {mean_soft_score:.2f}")
            else:
                print(f"{filename}: 'soft_score' column not found.")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

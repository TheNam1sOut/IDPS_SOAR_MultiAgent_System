import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import glob
import os

def clean_label(label):
    label = label.strip()
    label = label.replace('�', '-')   # Replace the faulty character
    return label

csv_files = glob.glob("./data/MachineLearningCSV/MachineLearningCVE/*.csv")
dst_dir = "./data/dataset"
os.makedirs(dst_dir, exist_ok=True)

df_list = []
for f in csv_files:
    df = pd.read_csv(f)
    df_list.append(df)

data = pd.concat(df_list, ignore_index=True)
data.columns = data.columns.str.strip() # Avoid errors like "Label   " instead of "Label" as expected
data['Label'] = data['Label'].apply(clean_label) # Label cleaning

print("Dataset shape:", data.shape)

labels = data['Label'].unique()
print("Labels in dataset:")
for l in labels:
    print(l)

# Saves the dataset
data.to_csv("./data/dataset/CleanlyLabelled_Dataset.csv", index=False)


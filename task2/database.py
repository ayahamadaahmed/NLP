import pandas as pd
import os

def save_to_csv(text, label, db_file="database.csv"):
    df = pd.DataFrame([[text, label]], columns=["Input", "Classification"])
    header = not os.path.exists(db_file)
    df.to_csv(db_file, mode="a", header=header, index=False)

def load_database(db_file="database.csv"):
    if os.path.exists(db_file):
        return pd.read_csv(db_file)
    else:
        return pd.DataFrame(columns=["Input", "Classification"])

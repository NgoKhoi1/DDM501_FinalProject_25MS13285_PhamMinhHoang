import pandas as pd
from sklearn.datasets import load_wine
import os

def generate_reference_data():
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    wine = load_wine()
    df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    df.to_csv(os.path.join(os.path.dirname(__file__), 'reference_data.csv'), index=False)
    print("Reference data generated at reference_data.csv")

if __name__ == "__main__":
    generate_reference_data()

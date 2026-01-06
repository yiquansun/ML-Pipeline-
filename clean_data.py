import pandas as pd

def clean_census_data(file_path, output_path):
    # Load data
    # skipinitialspace=True handles the most common issue in this dataset
    df = pd.read_csv(file_path, skipinitialspace=True)
    
    # Remove all remaining spaces from string columns just in case
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    # Save the clean version
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    clean_census_data("data/census.csv", "data/census_clean.csv")
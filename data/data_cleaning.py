import pandas as pd

def process_data(file_path):
    # skipinitialspace=True handles most spaces immediately
    df = pd.read_csv(file_path, skipinitialspace=True)
    
    # Strip any remaining whitespace from object (string) columns
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    # Save the cleaned version
    df.to_csv("data/census_clean.csv", index=False)
    print("Data cleaned and saved to data/census_clean.csv")

if __name__ == "__main__":
    process_data("data/census.csv")
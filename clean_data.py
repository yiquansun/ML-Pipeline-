import pandas as pd


def clean_census_data(path):
    """
    Remove leading and trailing spaces from string columns.
    """
    df = pd.read_csv(path)
    # Strip whitespace from all object columns
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    return df


if __name__ == "__main__":
    # Example execution: clean_census_data('data/census.csv')
    pass

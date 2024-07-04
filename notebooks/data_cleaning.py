import numpy as np
import pandas as pd

#### Functions 1. Data quality: A. Completeness and Missing values
## Functions
# replace unknown with np.nan
def replace_unknown_object_cols(df: pd.DataFrame, na_values:list = ["unknown"]) -> pd.DataFrame:
    """
    Replaces values "unknown" (or supplied values) with np.NaN in object and string-type columns of a DataFrame.
    Args:
        df (pandas.DataFrame): The DataFrame to modify.
    Returns:
        pd.DataFrame: The DataFrame with the replaced values.
    """
    object_cols = df.select_dtypes(include=[object,"string"]).columns
    default_value = ["unknown"]
    if na_values != default_value:
        na_values = na_values + default_value
    na_values_lower = [x.lower() for x in na_values]
    pattern = "|".join(na_values_lower)
    for col in object_cols:
        df[col] = df[col].str.strip().str.lower()
        df[col] = df[col].replace(pattern, np.nan, regex=True)
    return df

# assess prevalence of missingness
def prevalence_missingness(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the percentage of missing values per column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.Series: A Series with column names as the index and the percentage of missing values as values.
    """
    missing_percentage = round(df.isnull().sum() / len(df), 4) * 100
    missing_percentage = missing_percentage.reset_index().copy()
    missing_percentage = missing_percentage.rename(columns={"index":"col_name",0:"prevalence_na"})
    return missing_percentage

#### Functions 2. Data quality: B. Consistency in data formatting
## Functions

# convert to bool those columns whose values are [0,1] or [f,t]
def convert_bool_cols(df:pd.DataFrame) -> pd.DataFrame:
    """Converts columns in a DataFrame to bool if their unique values are [0, 1, np.nan].
    Args:
        df: A pandas DataFrame.
    Returns:
        The DataFrame with qualifying columns converted to bool.
    """
    # for [f,t]
    object_cols = df.select_dtypes(include=[object,"string"]).columns
    for col in df[object_cols].columns:
        df[col] = df.loc[:,col].str.lower().copy()
        if set(df[col].unique()) <= {"f","t", np.nan}:
            df[col] = df.loc[:,col].map({"f": 0, "t": 1}).copy()
    # for [0,1]
    for col in df.columns:
        if set(df[col].unique()) <= {0, 1, np.nan}:
            df[col] = df[col].astype(bool).copy()
    return df

# convert string columns matching 'date' to pd.datetime
def convert_obj_to_date(df:pd.DataFrame) -> pd.DataFrame:
    """Converts object columns containing dates to datetime format in a pandas DataFrame.
    Args:
    df (pandas.DataFrame): The DataFrame to convert.
    Returns:
    pandas.DataFrame: The DataFrame with object columns converted to datetime.
    """
    cols_date = [col for col in df.columns if 'date' in col]
    for col in df[cols_date].select_dtypes(include=[object,"string"]):
        try:
            df[col] = pd.to_datetime(df[col]).copy()
        except:
            pass
    return df

# convert string columns to category
def convert_to_category(df:pd.DataFrame, nunique_cutoff:int = 10) -> pd.DataFrame:
    """Converts string columns to pandas category if the column has less than 10 unique values.
    Args:
    df (pandas.DataFrame): The DataFrame to convert.
    Returns:
    pandas.DataFrame: The DataFrame with string columns converted to category if applicable.
    """
    for col in df.select_dtypes(include=[object,"string"]):
        if df[col].nunique() <= nunique_cutoff:
            df[col] = df[col].astype("category")
    return df

# function that avoids repeated column names when joining aggregated dataframes
def append_suffix(df:pd.DataFrame, columns:list, suffix:str) -> pd.DataFrame:
    """
    Appends a suffix to the names of selected columns in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        columns (list): A list of column names to modify.
        suffix (str): The suffix to append.

    Returns:
        pandas.DataFrame: A new DataFrame with modified column names.
    """

    new_cols = [col + suffix for col in columns if col in df.columns]
    return df.rename(columns=dict(zip(columns, new_cols)))
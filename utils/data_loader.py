import pandas as pd

def loadDataset(data_path="digitalizações_registadas.csv", 
                date_columns=[],
                columns_to_drop_with_null=[],
                number_columns=[],
                delimiter=";",
                date_format=""):
    # Load data 
    dataframe = pd.read_csv(data_path, 
                        delimiter=delimiter,
                        date_format=date_format, 
                        parse_dates=date_columns) 

    # Strip any leading or trailing whitespace from column names
    dataframe.columns = dataframe.columns.str.strip()

    # Get unnamed columns to remove
    unnamed_columns = [col for col in dataframe.columns if col.startswith('Unnamed')]

    # Drop unnamed columns
    dataframe = dataframe.drop(columns=unnamed_columns)

    # Drop rows with any null values
    if len(columns_to_drop_with_null) > 0:
        dataframe = dataframe.dropna(subset=columns_to_drop_with_null)

    # TODO: REMOVE THIS. THIS IS ONLY HERE FOR DEMO DATA: Drop specific row with date in year 221
    dataframe = dataframe[dataframe['Nº documento Fornecedor'] != "ZRF2 2/6001001951"]        

    # Convert columns to date type
    for date_col in date_columns:
        dataframe[date_col] = pd.to_datetime(dataframe[date_col], format="%d/%m/%Y")

    # Convert num columns to number or float
    for num_col in number_columns:
        dataframe[num_col] = dataframe[num_col].str.replace(',', '').astype(float)
    
    return dataframe


def add_count(main_df, supplier_col_main, file_path, supplier_col_file, new_col_name, feature_columns, delimiter=";"):
    """
    Adds supplier count from a specified file to the main dataframe.

    Parameters:
    - main_df: The main dataframe to which the count will be added.
    - supplier_col_main: The column name in the main dataframe that contains supplier names.
    - file_path: File path to the CSV file containing supplier data.
    - supplier_col_file: The column name in the CSV file that contains supplier names.
    - new_col_name: The name of the new count column to be added to the main dataframe.
    - delimiter: The delimiter used in the CSV file (default is ";").

    Returns:
    - Modified dataframe with the new count column added.
    """
    
    # Load data
    df = pd.read_csv(file_path, delimiter=delimiter)
    
    # Calculate counts
    counts = df[supplier_col_file].value_counts().to_dict()
    
    # Map counts to the main dataframe
    main_df[new_col_name] = main_df[supplier_col_main].map(counts).fillna(0).astype(int)

    feature_columns.append(new_col_name)
    
    return main_df
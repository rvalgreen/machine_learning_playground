import pandas as pd

def loadDataset(data_path="digitalizações_registadas.csv", 
                date_columns=[],
                columns_to_drop_with_null=[],
                number_columns=[],
                delimiter=";",
                date_format=""):
    # Load data 
    dataFrame = pd.read_csv(data_path, 
                        delimiter=delimiter,
                        date_format=date_format, 
                        parse_dates=date_columns) 

    # Strip any leading or trailing whitespace from column names
    dataFrame.columns = dataFrame.columns.str.strip()

    # Get unnamed columns to remove
    unnamed_columns = [col for col in dataFrame.columns if col.startswith('Unnamed')]

    # Drop unnamed columns
    dataFrame = dataFrame.drop(columns=unnamed_columns)

    # Drop rows with any null values
    if len(columns_to_drop_with_null) > 0:
        dataFrame = dataFrame.dropna(subset=columns_to_drop_with_null)

    # Convert columns to date type
    for date_col in date_columns:
        dataFrame[date_col] = pd.to_datetime(dataFrame[date_col], format="%d/%m/%Y")

    # Convert num columns to number or float
    for num_col in number_columns:
        dataFrame[num_col] = dataFrame[num_col].str.replace(',', '').astype(float)


def add_count(main_df, supplier_col_main, file_path, supplier_col_file, new_col_name, delimiter=";"):
    """
    Adds supplier count from a specified file to the main DataFrame.

    Parameters:
    - main_df: The main DataFrame to which the count will be added.
    - supplier_col_main: The column name in the main DataFrame that contains supplier names.
    - file_path: File path to the CSV file containing supplier data.
    - supplier_col_file: The column name in the CSV file that contains supplier names.
    - new_col_name: The name of the new count column to be added to the main DataFrame.
    - delimiter: The delimiter used in the CSV file (default is ";").

    Returns:
    - Modified DataFrame with the new count column added.
    """
    
    # Load data
    df = pd.read_csv(file_path, delimiter=delimiter)
    
    # Calculate counts
    counts = df[supplier_col_file].value_counts().to_dict()
    
    # Map counts to the main DataFrame
    main_df[new_col_name] = main_df[supplier_col_main].map(counts).fillna(0).astype(int)
    
    return main_df
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


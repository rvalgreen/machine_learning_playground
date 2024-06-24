import pandas as pd

# Function to format the row data
def format_row(row, feature_columns, dataframe):
    parts = []
    for column in feature_columns:
        value = row[column]
        if pd.api.types.is_datetime64_any_dtype(dataframe[column]):
            value = value.strftime('%d/%m/%Y')
        parts.append(f"{column}:{value}")
    return '\n '.join(parts) + "\n"


def buildDocRepresentation(dataframe, feature_columns):
    dataframe['FullText'] = dataframe.apply(lambda row: format_row(row, feature_columns, dataframe), axis = 1)

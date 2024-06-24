import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import torch
from transformers import DataCollatorWithPadding, GPT2Tokenizer, DistilBertForSequenceClassification, DistilBertModel, DistilBertTokenizer, TrainingArguments, Trainer, TrainingArguments
from datasets import load_metric
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc,  mean_absolute_error, mean_squared_error, r2_score
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import random

from utils.data_loader import loadDataset
from utils.params_parser import ParamsParser


if __name__ == '__main__':
    # Parse script args
    parser_wrapper = ParamsParser()
    parser = parser_wrapper.getParser()
    args = parser.parse_args()
    print("###### ARGS ######")
    print(args)
    print("###### ###### ######")

    # Set seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
    np.random.seed(seed)
    random.seed(seed)


    # TODO: logic for model saving. directories, etc
    save = False
    save_dir="models/"+args.runName

    # Load data
    dataframe = []

    # TODO: send this to args parser?
    date_columns = ['Data Emissão','Data entrada','Data vencimento indicada']
    columns_to_drop_with_null=['Data vencimento indicada','Data Emissão','Origem']
    column_label = "Origem"
    feature_columns = ["Fornecedor","Data Emissão","Data entrada","Data vencimento indicada", "Valor com IVA"]
    number_columns = ['Valor com IVA']
    string_columns = ["Fornecedor"]

    # load dataset via pandas dataframe
    dataframe = loadDataset(data_path=args.dataset,
                       date_columns=date_columns,
                       number_columns=number_columns,
                       columns_to_drop_with_null=columns_to_drop_with_null)
    


    # TODO: remove this. this is only here bc i was lazy and havent exported
    # the data from RM correctly (without Requisição)
    dataframe = dataframe[dataframe['Origem'] != "Requisição"]

    # Set Labels column (this is unecessary as we can use Origem - but good for readability)
    dataframe['Labels'] = dataframe[column_label]


    # Feature Engineering
    for date_col in date_columns:
        month_col_name = date_col+"_month"
        weekday_col_name = date_col+"_weekday"
        dataframe[month_col_name] = dataframe[date_col].dt.month
        dataframe[weekday_col_name] = dataframe[date_col].dt.dayofweek


    # Init label encoder
    label_encoder = LabelEncoder()

    # TODO: send this to param args
    timesplit = True
    timesplit_date = '2024-02-01'
    timesplit_column = "Data entrada"    

    if timesplit:
        # Split data according to specified date
        dataframe_before = dataframe[dataframe[timesplit_column] < timesplit_date]
        dataframe_after = dataframe[dataframe[timesplit_column] >= timesplit_date]

        # Encode labels
        y_train = label_encoder.fit_transform(dataframe_before['Labels'].tolist())
        y_test = label_encoder.fit_transform(dataframe_after['Labels'].tolist())

        # Drop unused / unwanted columns/features
        dataframe_before = dataframe_before.drop(
            columns=[col for col in dataframe_before.columns if col not in feature_columns], axis=1
        )

        # Drop unused / unwanted columns/features
        dataframe_after = dataframe_after.drop(
            columns=[col for col in dataframe_after.columns if col not in feature_columns], axis=1
        )

        # Encode string columns
        for col in string_columns:
            dataframe_before[col] = label_encoder.fit_transform(dataframe_before[col])
            dataframe_after[col] = label_encoder.fit_transform(dataframe_after[col])

    else:
        # Encode string columns
        for col in string_columns:
            dataframe[col] = label_encoder.fit_transform(a[col])

        y = label_encoder.fit_transform(dataframe['Labels'])
        X_train, X_test, y_train, y_test = train_test_split(dataframe, y, test_size=0.2, random_state=args.seed)

    
    # Fit model
    model = RandomForestClassifier(n_estimators=100, random_state=args.seed)
    model.fit(X_train, y_train)    


    # Eval
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    print('Mean Absolute Error:', mae)


    # Calculate MSE
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error (MSE): {mse}')

    # Calculate R-squared
    r2 = r2_score(y_test, predictions)
    print(f'R-squared (R²): {r2}')


    # Calculate accuracy
    accuracy = sum(1 for true, pred in zip(y_test, predictions) if true == pred) / len(y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')
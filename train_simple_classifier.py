import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
import torch
from transformers import DataCollatorWithPadding, GPT2Tokenizer, DistilBertForSequenceClassification, DistilBertModel, DistilBertTokenizer, TrainingArguments, Trainer, TrainingArguments
from datasets import load_metric
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc,  mean_absolute_error, mean_squared_error, r2_score
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import random
import joblib
from utils.data_loader import loadDataset, add_count
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
    


    # TODO: REMOVE THIS. THIS IS ONLY HERE FOR DEMO DATA: Drop specific row with date in year 221
    dataframe = dataframe[dataframe['Nº documento Fornecedor'] != "ZRF2 2/6001001951"]

    # TODO: remove this. this is only here bc i was lazy and havent exported
    # the data from RM correctly (without Requisição)
    dataframe = dataframe[dataframe['Origem'] != "Requisição"]

    # Set Labels column (this is unecessary as we can use Origem - but good for readability)
    dataframe['Labels'] = dataframe[column_label]


    # Equal sample size 
    def select_by_date(group):
        return group.sort_values(by='Data entrada').head(min_size)

    if args.equalSample:
        grouped = dataframe.groupby('Labels')
        min_size = grouped.size().min()
        dataFrame = grouped.apply(lambda x: x.sample(min_size)).reset_index(drop=True)

    # Feature Engineering
    for date_col in date_columns:
        month_col_name = date_col+"_month"
        weekday_col_name = date_col+"_weekday"
        day_col_name=date_col+"_day"
        dataframe[month_col_name] = dataframe[date_col].dt.month
        dataframe[weekday_col_name] = dataframe[date_col].dt.dayofweek
        dataframe[day_col_name] = dataframe[date_col].dt.day


    # Init label encoder
    label_encoder = LabelEncoder()

    # TODO: send this to param args
    timesplit = True
    timesplit_date = '2024-02-01'
    timesplit_column = "Data entrada"    

    ### Insert here somehow extra data processing / feature engineering steps ###
    # Adding count from requisition file
    dataframe = add_count(
        main_df=dataframe, 
        supplier_col_main="Fornecedor", 
        file_path='documentos_req.csv', 
        supplier_col_file='Nome Fornecedor', 
        new_col_name='N Requisiçoes Fornecedor'
    )

    # Adding count from contract file
    dataframe = add_count(
        main_df=dataframe, 
        supplier_col_main="Fornecedor", 
        file_path='contratos.csv', 
        supplier_col_file='Nome Fornecedor', 
        new_col_name='N Contratos Fornecedor'
    )
    ######

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

        X_train = dataframe_before
        X_test = dataframe_after
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
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)


    # Draw plots and save them
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.1)
    plt.xlabel('Actual Value of dataFrame')
    plt.ylabel('Predicted Value of dataFrame')
    plt.title('Actual vs Predicted Values of dataFrame')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./plots/actual_vs_predicted_plot.png', format='png') 


    # Compute the confusion matrix
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    plt.savefig('./plots/confusion_matrix.png', format='png')


    # Feature Importance
    feature_importance = model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print('Feature Importance:')
    print(feature_importance_df)


    # Save the model to a file
    save_name = args.wandb+"_model.pkl"
    joblib.dump(model, save_name)

    # Load the model from the file
    # loaded_rfc = joblib.load(save_name)

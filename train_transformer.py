import pandas as pd
from dotenv import load_dotenv
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW,DataCollatorWithPadding, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, BertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments,AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from datasets import Dataset as HFDataset, load_metric
from datasets import load_dataset
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
import wandb
from utils.encoded_dataset import EncodedDataset
from utils.optimizer_scheduler import getOptimizer
from utils.params_parser import ParamsParser
from utils.data_loader import loadDataset
from utils.doc_text_representation import buildDocRepresentation
import os

if __name__ == '__main__':
    # Parse script args
    parser_wrapper = ParamsParser()
    parser = parser_wrapper.getParser()
    args = parser.parse_args()
    print("###### ARGS ######")
    print(args)
    print("###### ###### ######")

    # Load env vars if necessary
    load_dotenv("wandb.env")

    # Login to wandb using api key
    wandb.login(key=os.getenv("WANDB_API_KEY"))

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
    date_columns = ['Data Emissão','Data vencimento indicada',"Data entrada"]
    columns_to_drop_with_null=['Data vencimento indicada','Data Emissão','Origem']
    column_label = "Origem"
    feature_columns = ["Fornecedor","Data Emissão","Data entrada","Data vencimento indicada", "Valor com IVA"]

    # load dataset via pandas dataframe
    dataframe = loadDataset(data_path=args.dataset,
                       date_columns=date_columns,
                       columns_to_drop_with_null=columns_to_drop_with_null)
    

    # TODO: remove this. this is only here bc i was lazy and havent exported
    # the data from RM correctly (without Requisição)
    dataframe = dataframe[dataframe['Origem'] != "Requisição"]

    # Set Labels column (this is unecessary as we can use Origem - but good for readability)
    dataframe['Labels'] = dataframe[column_label]

    # Build column with our document's text representation
    buildDocRepresentation(dataframe, feature_columns)

    # Init tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(args.model)
    special_tokens_dict = {"pad_token": "<PAD>"}
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(special_tokens_dict)

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

        # Especify what is train/test for readability
        train_texts = dataframe_before['FullText'].tolist()
        test_texts = dataframe_after['FullText'].tolist()
        train_labels = dataframe_before['Labels'].tolist()
        test_labels = dataframe_after['Labels'].tolist()

        # Encode labels - model cant take actual text - we need to encode text to numbers
        train_labels = label_encoder.fit_transform(train_labels)
        test_labels = label_encoder.fit_transform(test_labels)

        # Encode our document text representations
        train_texts_encoded = tokenizer(train_texts, truncation=True, padding=True, max_length=128 )
        test_texts_encoded = tokenizer(test_texts, truncation=True, padding=True, max_length=128 )

        # Send encoded data to pytorch dataset 
        train_dataset = EncodedDataset({'input_ids': train_texts_encoded['input_ids'], 
                                        'attention_mask': train_texts_encoded['attention_mask']}, train_labels)
        test_dataset = EncodedDataset({'input_ids': test_texts_encoded['input_ids'], 
                                       'attention_mask': test_texts_encoded['attention_mask']}, test_labels)

    else:
        texts = dataframe['FullText'].tolist()
        labels = dataframe['Labels'].tolist()
        # Encode labels
        encoded_labels = label_encoder.fit_transform(labels)

        encodings = tokenizer(texts, truncation=True, padding=True, max_length=128 )
        # Split dataset into training and test sets
        train_texts_encoded, test_texts_encoded, train_labels, test_labels = train_test_split(
            encodings['input_ids'], encoded_labels, test_size=0.15, random_state=seed
        )

        train_masks, val_masks = train_test_split(
            encodings['attention_mask'], test_size=0.15, random_state=seed
        )

        # Send encoded data to pytorch dataset 
        train_dataset = EncodedDataset({'input_ids': train_texts_encoded, 'attention_mask': train_masks}, train_labels)
        test_dataset = EncodedDataset({'input_ids': test_texts_encoded, 'attention_mask': val_masks}, test_labels)


    # Prepare dataloaders
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)

    # Params (for readability)
    epochs = args.epochs
    lr = args.learningRate
    wd = args.weightDecay
    ws = args.warmupSteps

    # Train model
    model = DistilBertForSequenceClassification.from_pretrained(args.model, num_labels=len(label_encoder.classes_))

    # move model to device
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    model.to(device)

    # Calculate total train steps for scheduler
    num_training_steps = len(train_loader) * epochs

    # Init optimizer and scheduler
    optimizer, scheduler = getOptimizer(model, num_training_steps, scheduler_type=args.scheduler,
                                        lr=lr, weight_decay=wd,
                                        warmup_steps=ws)
    
    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
    train_losses = []
    steps = 0


    # Wandb init conf
    runName = f"experiment_{args.runName}"
    run = wandb.init(
        project="cob-demo",
        name=runName, 
        config={
            args
        },
    )


    # Train model
    for epoch in range(epochs):
        epoch_train_loss = 0.0
        for batch in train_loader:
            steps += 1

            # get inputs
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # forward inputs through model
            outputs = model(input_ids, attention_mask)
            # calculate loss
            loss = criterion(outputs.logits, labels)
            # perform backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_train_loss += loss.item()
            wandb.log({"loss": loss.item(), "step":steps, "learning_rate":scheduler.get_last_lr()[0]})


        # Calculate average training loss for the epoch
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        print(f"Run: {args.runName}   |Epoch {epoch + 1}, Train Loss: {epoch_train_loss}")


        # Evaluate model
        model.eval()

        val_loss = 0.0
        correct = 0
        total = 0
        val_losses = []
        val_accuracies = []
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                #outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                outputs = model(input_ids, attention_mask)

                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs.logits, labels)

                val_loss += loss.item()
                    
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        # Calculate average validation loss and accuracy
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        
        print(f"Validation Loss: {val_loss}, Accuracy: {accuracy}%")        



    report = classification_report(y_true, y_pred, output_dict=True)
    print("Classification Report:")
    print(report)

    wandb.run.summary["report"] = report
    wandb.finish()
    
    if save:
        model.save_pretrained(save_dir)
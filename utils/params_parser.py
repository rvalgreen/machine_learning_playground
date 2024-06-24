import argparse
import os
import json
from pathlib import Path

class ParamsParser():


    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Model training script')

        # Training Parameters
        self.parser.add_argument('-b', dest='batchSize', type=int, default=32, help='Training batch size')
        self.parser.add_argument('-lr', dest='learningRate', default="1e-4", help='Learning rate')
        self.parser.add_argument('-wd', dest='weightDecay', type=float, default=0, help='Weight decay')
        self.parser.add_argument('-ws', dest='warmupSteps', type=float, default=0, help='Warmup steps')
        self.parser.add_argument('-ml', dest='maxLength', type=int, default=128, help='Max sequence length')
        self.parser.add_argument('-e', dest='epochs', type=int, default=1, help='Training epochs')

        # Data
        self.parser.add_argument('-d', dest='dataset', default="digitalizações_registadas.csv", help='Dataset/file to use')

        # Model
        self.parser.add_argument('-t', dest='tokenizer', default="distilbert-base-uncased", help='Tokenizer to use')
        self.parser.add_argument('-m', dest='model', default="distilbert-base-uncased", help='Base model to use')

        # Optim + Scheduler
        self.parser.add_argument('-scheduler', dest='scheduler', default="cosine", help='Scheduler')
        self.parser.add_argument('-optimizer', dest='optimizer', default="adamw", help='Optimizer')
        
        # Other
        self.parser.add_argument('-seed', dest='seed', type=int, default=42,
                                 help='Training seed', required=False)
        self.parser.add_argument('-rn', dest='runName', type=int, default="my_run",
                                 help='Wandb run name', required=False)
        
        # Wandb
        self.parser.add_argument('-wandb', dest='wandb', default="no-name", help='Wandb run name')
        self.parser.add_argument('-wandb_env', dest='wandb_env', default="wandb_gpt.env", help='Wandb env file')

    
    def getParser(self):
        return self.parser  
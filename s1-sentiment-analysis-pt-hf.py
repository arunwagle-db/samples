# Databricks notebook source
# Only required if not ML Cluster
# !pip install transformers
# !pip install tensorflow

# COMMAND ----------

# MAGIC %md
# MAGIC Check the number of devices available

# COMMAND ----------

from tensorflow.python.client import device_lib
import multiprocessing
print(multiprocessing.cpu_count())
device_lib.list_local_devices()

# COMMAND ----------

# MAGIC %md
# MAGIC Get the data

# COMMAND ----------

!wget https://dem-primary-tweets.s3.amazonaws.com/PeteForAmerica.1574004110.txt

# COMMAND ----------

# MAGIC %md
# MAGIC Copy the data to locally to dbfs

# COMMAND ----------

# MAGIC %fs
# MAGIC 
# MAGIC cp file:/Workspace/Repos/arun.wagle@databricks.com/samples/PeteForAmerica.1574004110.txt dbfs:/Users/arun.wagle@databricks.com/Pete2.txt

# COMMAND ----------

import pandas as pd
df = pd.read_json('/dbfs/Users/arun.wagle@databricks.com/Pete2.txt', lines=True)
df.head(5)

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn

MODEL =  "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

tokenized_text = tokenizer(
["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt")

outputs = model(**tokenized_text)
predictions = nn.functional.softmax(outputs.logits, dim=1)
predictions


# COMMAND ----------

# MAGIC %md
# MAGIC Helper functions

# COMMAND ----------

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import csv
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import urllib
import json
import glob
import os
from torch.utils.data import Dataset

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

def get_all_files():
  file_list = ['/dbfs/Users/arun.wagle@databricks.com/Pete2.txt']
  return(file_list)


class TextLoader(Dataset):
    def __init__(self, file=None, transform=None, target_transform=None, tokenizer=None):
        self.file = pd.read_json(file, lines=True)
        self.file = self.file
        self.file = tokenizer(list(self.file['full_text']), padding=True, truncation=True, max_length=512, return_tensors='pt')
        self.file = self.file['input_ids']
        self.transform = transform
        self.target_transform = target_transform
 
    def __len__(self):
        return len(self.file)
 
    def __getitem__(self, idx):
        data = self.file[idx]
        return(data)
      
class SentimentModel(nn.Module):
    # Our model
 
    def __init__(self):
        super(SentimentModel, self).__init__()
        #print("------------------- Initializing once ------------------")
        self.fc = AutoModelForSequenceClassification.from_pretrained(MODEL)
 
    def forward(self, input):
        #print(input)
        output = self.fc(input)
        pt_predictions = nn.functional.softmax(output.logits, dim=1)
        #print("\tIn Model: input size", input.size())
        return(pt_predictions)
      

# COMMAND ----------

dev = 'cuda'
if dev == 'cpu':
  device = torch.device('cpu')
  device_staging = 'cpu:0'
else:
  device = torch.device('cuda')
  device_staging = 'cuda:0'
  
tokenizer = AutoTokenizer.from_pretrained(MODEL)
 
all_files = get_all_files()
model = SentimentModel()
try:
      # If you leave out the device_ids parameter, it selects all the devices (GPUs) available
      #     device_ids=device_ids
      model = nn.DataParallel(model) 
      model.to(device_staging)
except:
      torch.set_printoptions(threshold=10000)

    
for file in all_files:
    data = TextLoader(file=file, tokenizer=tokenizer)
    train_dataloader = DataLoader(data, batch_size=120, shuffle=False) # Shuffle should be set to False
    out = torch.empty(0,0)
    for data in train_dataloader:
        input = data.to(device_staging)
        #print(len(input))
        if(len(out) == 0):
          out = model(input)
        else:
          output = model(input)
          with torch.no_grad():
            out = torch.cat((out, output), 0)
            
    df = pd.read_json(file, lines=True)['full_text']
    res = out.cpu().numpy()
    df_res = pd.DataFrame({ "text": df, "negative": res[:,0], "positive": res[:,1]})
    print(df_res)    

# Databricks notebook source
# MAGIC %md
# MAGIC 1. Login to https://console.cloud.google.com/ . Enable Gmail API's
# MAGIC 2. Create "Credentials". Create "Service Accounts" > Databricks-Notebook-Svc-Acct. 
# MAGIC 3. Select "Databricks-Notebook-Svc-Acct" > Add Key . This will save json on your local machine.  
# MAGIC 2. https://admin.google.com/
# MAGIC   a. Go to Security > API Controls > Settings. Enable "Trust internal, domain-owned apps"
# MAGIC   b. Click on "Manage Domain-wide Delegation". 
# MAGIC   c. Input the Client ID* (you can get this from the file downloaded in step 3 above) of your service account and in OAuth scopes input: https://www.googleapis.com/auth/gmail.readonly 
# MAGIC   

# COMMAND ----------

!pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

# COMMAND ----------

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# COMMAND ----------

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
SERVICE_ACCOUNT_FILE = '/dbfs/FileStore/Users/arun.wagle@databricks.com/inceptopia_service_account.json'

credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

delegated_credentials = credentials.with_subject('arun.wagle@inceptopia.com')

# COMMAND ----------

def read_all_labels():
  try:
      # Call the Gmail API
      service = build('gmail', 'v1', credentials=delegated_credentials)
      print (service)
      results = service.users().labels().list(userId='me').execute()
      labels = results.get('labels', [])

      if not labels:
          print('No labels found.')
          return
      print('Labels:')
      for label in labels:
          print(label['id'])
          print(label['name'])

  except HttpError as error:
      # TODO(developer) - Handle errors from gmail API.
      print(f'An error occurred: {error}')
      
      
def get_label_ids(search_labels):
  try:
      label_ids=[]
      print("search_label:: {}".format(search_labels))
      # Call the Gmail API
      service = build('gmail', 'v1', credentials=delegated_credentials)
      print (service)
      results = service.users().labels().list(userId='me').execute()
      labels = results.get('labels', [])

      if not labels:
          print('No labels found.')
          return None      
      for label in labels:
          name = label['name']
          if name in search_labels:
            label_ids.append(label['id'])
            
      return label_ids
            
  except HttpError as error:
      # TODO(developer) - Handle errors from gmail API.
      print(f'An error occurred: {error}')      

# COMMAND ----------

def read_all_messages_for_label(label_names):
  try:
      # Call the Gmail API
      label_ids = get_label_ids (label_names)
      print ("label_ids::{}".format(label_ids))
      service = build('gmail', 'v1', credentials=delegated_credentials)
      
      messages_inst = service.users().messages()
      results = messages_inst.list(userId='me', labelIds=label_ids).execute()
      messages = results.get('messages', [])

      if not messages:
          print('No messages found.')
          return
      for message in messages:
          print(message)
          id = message['id']
          thread_id = message['threadId']
          
          results = messages_inst.get(userId='me', id=id, format='full').execute()
          print (results)
#           message_details = results.get('messages', [])
#           if message_details:
#             print (message_details)
          

  except HttpError as error:
      # TODO(developer) - Handle errors from gmail API.
      print(f'An error occurred: {error}')

# COMMAND ----------

label_names = ["Insurance"]
read_all_messages_for_label(label_names)
# read_all_labels()

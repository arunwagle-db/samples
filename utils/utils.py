# Databricks notebook source
dbutils.help()

# COMMAND ----------

dbutils.notebook.help()

# COMMAND ----------

# print_ctx_tags()
# user_home_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("user_home_path").getOrElse(None)
# print ("user_home_path::{}".format(user_home_path))

current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("user").getOrElse(None)
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().getOrElse(None)
cluster_id = dbutils.notebook.entry_point.getDbutils().notebook().getContext().clusterId().getOrElse(None)
api_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
api_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

print ("user::{}".format(current_user))
print ("notebook_path::{}".format(notebook_path))
print ("cluster_id::{}".format(cluster_id))
print ("api_url::{}".format(api_url))
print ("api_token::{}".format(api_token))

print (dbutils.notebook.entry_point.getDbutils().notebook())

# COMMAND ----------

# dbfs mkdirs dbfs:/FileStore/first.last@databricks.com
# dbutils.fs.mv('dbfs:/Home/arun.wagle@databricks.com', 'dbfs:/Home/arun.wagle@databricks.com',True) # `True` for recursive
dbutils.fs.help()

# COMMAND ----------

# dbutils.fs.mkdirs('dbfs:/home/arun.wagle@databricks.com/images/tables')

# dbutils.fs.ls('dbfs:/databricks-datasets')


# dbutils.fs.mv("dbfs:/FileStore/arunwagle/tables", "dbfs:/home/arun.wagle@databricks.com/images/tables", recurse=True)
dbutils.fs.rm("dbfs:/home/arun.wagle@databricks.com/data/checkpoint", recurse=True)
dbutils.fs.rm("dbfs:/home/arun.wagle@databricks.com/data/delta", recurse=True)

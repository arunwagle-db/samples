-- Databricks notebook source
CREATE CATALOG IF NOT EXISTS arun_wagle_bizai_catalog COMMENT 'This is catalog for business flow automation';

USE CATALOG arun_wagle_bizai_catalog; 

--- Grant create schema, create table, create view, & use catalog permissions to all users on the account
--- This also works for other account-level groups and individual users
GRANT CREATE SCHEMA, CREATE TABLE, CREATE VIEW, USE CATALOG
ON CATALOG arun_wagle_bizai_catalog
TO `arun.wagle@databricks.com`;

--- Check grants on the quickstart catalog
SHOW GRANT ON CATALOG arun_wagle_bizai_catalog;

-- COMMAND ----------

CREATE SCHEMA IF NOT EXISTS bizaischema COMMENT 'This schema is created by Arun Wagle and used for automating business flows' ;

-- Describe a schema
DESCRIBE SCHEMA bizaischema;

-- Set the current schema
USE bizaischema;

-- COMMAND ----------

-- PARTITIONED BY (columnA);PRIMARY KEY(first_name, last_name);FOREIGN KEY(owner_first_name, owner_last_name) REFERENCES persons
-- GENERATED ALWAYS AS IDENTITY (START WITH 1, INCREMENT BY 1, NO CACHE )

-- ROOT_FOLDER = 'dbfs:/home/arun.wagle@databricks.com'
-- SOURCE_IMAGE_FOLDER = "{}{}".format(ROOT_FOLDER, TABLES_IMAGE_FOLDER)
-- DELTA_TABLE_PATH = "{}{}".format(ROOT_FOLDER, "/data/delta") 
-- Note: DEFAULKT values not supported for Delta tables in CREATE TABLE 
CREATE TABLE IF NOT EXISTS bizai_ref_country (
	id  BIGINT GENERATED ALWAYS AS IDENTITY,
	two_digit_country_code STRING  NOT NULL ,
	three_digit_country_code STRING  NOT NULL ,
	name STRING,	
	first_updated TIMESTAMP NOT NULL,
	last_updated TIMESTAMP NOT NULL,
    CONSTRAINT bizai_ref_country_pk PRIMARY KEY(id)
);
-- LOCATION 'dbfs:/home/arun.wagle@databricks.com/data/delta';

CREATE TABLE IF NOT EXISTS bizai_client_address (
	id  BIGINT GENERATED ALWAYS AS IDENTITY ,
	address STRING  NOT NULL ,
	state STRING  NOT NULL ,
	city STRING  NOT NULL ,
	country_id BIGINT  NOT NULL ,
	first_updated TIMESTAMP NOT NULL ,
	last_updated TIMESTAMP NOT NULL,
    CONSTRAINT bizai_client_address_pk PRIMARY KEY(id),
	CONSTRAINT country_id_fk FOREIGN KEY (country_id) REFERENCES bizai_ref_country(id)
);
-- LOCATION 'dbfs:/home/arun.wagle@databricks.com/data/delta';

CREATE TABLE IF NOT EXISTS bizai_client (
	id  BIGINT GENERATED ALWAYS AS IDENTITY ,
	parent_id BIGINT,
	entity_name STRING  NOT NULL ,
	legal_name STRING  NOT NULL ,		
	short_name STRING NOT NULL ,	
	address_id BIGINT NOT NULL,
	setup_complete BOOLEAN ,
	first_updated TIMESTAMP NOT NULL ,
	last_updated TIMESTAMP NOT NULL ,
	is_system_client BOOLEAN ,
    CONSTRAINT bizai_client_pk PRIMARY KEY(id),
-- 	CONSTRAINT bizai_client_self_fk FOREIGN KEY (parent_id) REFERENCES bizai_client(id),
	CONSTRAINT bizai_client_address_fk FOREIGN KEY (address_id) REFERENCES bizai_client_address(id)
);
-- LOCATION 'dbfs:/home/arun.wagle@databricks.com/data/delta';

CREATE TABLE IF NOT EXISTS bizai_client_engagement (
	id  BIGINT GENERATED ALWAYS AS IDENTITY ,
	name STRING NOT NULL,
	client_id BIGINT NOT NULL,
	description STRING,	
	doc_class STRING NOT NULL, -- CONTRACT
	first_updated timestamp NOT NULL,
	last_updated timestamp NOT NULL,
    CONSTRAINT bizai_client_id_fk FOREIGN KEY (client_id) REFERENCES bizaischema.bizai_client(id)
);


-- COMMAND ----------

-- Note you cannot have self contraints and CREATE TABLE IF NOT EXISTS in the same SQL and hence ALTER TABLE run seperately here
ALTER TABLE bizai_client ADD CONSTRAINT bizai_client_self_fk FOREIGN KEY (parent_id) REFERENCES bizai_client(id);

-- COMMAND ----------

-- View all tables in the schema
SHOW TABLES IN bizaischema;


-- COMMAND ----------

-- Describe this table
DESCRIBE TABLE EXTENDED bizai_client;

-- COMMAND ----------

-- INSERT STATEMENTS: END: Onboard default System client

INSERT
  INTO  bizai_ref_country (two_digit_country_code, three_digit_country_code, name, first_updated, last_updated)
  VALUES('US','USA', 'United States of America', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP());

-- INSERT
--   INTO  bizai_client_address ('address', 'state', 'city', 'country_id', 'first_updated', 'last_updated') 
--   VALUES('168 Price Ct', 'NJ','WEST NEW YORK',1, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP());

-- INSERT
--   INTO  bizai_client ('parent_id', 'entity_name', 'legal_name', 'short_name', 'address_id', 'first_updated', 'last_updated') 
--   VALUES(-1,'BizAI','BizAI','BizAI',1, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP());

-- COMMAND ----------

INSERT OVERWRITE bizai_client_address (address, state, city, country_id, first_updated, last_updated) 
  VALUES('168 Price Ct', 'NJ','WEST NEW YORK',1, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP());

INSERT OVERWRITE bizai_client (parent_id, entity_name, legal_name, short_name, address_id, setup_complete, is_system_client, first_updated, last_updated) 
  VALUES(-1,'BizAI','BizAI','BizAI',1, True, True, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP());
  
INSERT OVERWRITE bizai_client_engagement (name, client_id, description, doc_class, first_updated, last_updated ) 
  VALUES('TABLE MANAGEMENT',1,'TABLE MANAGEMENT USECASE','TABLES', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP());

-- COMMAND ----------

SELECT * from bizai_client;

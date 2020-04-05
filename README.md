# Disaster Response Pipeline Project

## Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#Instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>

This is one of the projects for the [Udacity Nanodegree Data Scientist](https://www.udacity.com/course/data-scientist-nanodegree--nd025). The project classifies disaster response messages into categories by using a machine learning pipeleline. A flask app allows you to type in a message and classify it. 

## File Descriptions<a name="files"></a>

The data was provided by Udacity and can be found in the data folder. 
- 'disaster_messages.csv' contains the messages 
- 'disaster_categories.csv' contains the coded categories of the messages

The data folder als contains 'process_data.py' which is the ETL pipeline transforming the csvs and storing them in a sql database.

The 'models' folder contains:
- train_classifier.py: the machine learning pipeline that loads the sql database and pickles a classifier
- 'ML Pipeline Preparation.ipynb': jupyter notebook used to prepare 'train_classifyer.py'
- my_tokenizer.py: tokenize function used by CountVectorizer in the machine learning pipeline.

The app folder contains the files needed to run the flask app:
- my_tokenizer.py: copy of 'my_tokenizer.py' from the models folder. 
- run.py: python script to run the app
- templates/ includes 'master.html' and 'go.html' which are the html pages used by the app. The first is the main page with some data visualisations and the latter shows which category the text that the user has typed in belongs to.

## Instructions<a name="Instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

    Note: the ML pipeline is based on 'ML Pipeline Preparation.ipynb', available in the 'models' folder.
    Note: the ML pipeline imports 'my_tokenizer.py'

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Feel free to use the code. 

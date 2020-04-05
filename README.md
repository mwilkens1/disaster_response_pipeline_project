# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Instructions](#Instructions)
4. [File Descriptions](#files)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>

This is one of the projects for the [Udacity Nanodegree Data Scientist](https://www.udacity.com/course/data-scientist-nanodegree--nd025). The project classifies disaster response messages into categories by using a machine learning pipeleline. A flask app allows you to type in a message and classify it. 

### Instructions<a name="Instructions"></a>

The data was provided by Udacity and can be found in the data folder. 'Disaster_messages.csv' contains the messages and 'disaster_categories.csv' contains the coded categories.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

    Note: the ML pipeline is based on 'ML Pipeline Preparation.ipynb', available in the 'models' folder.

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Feel free to use the code. 

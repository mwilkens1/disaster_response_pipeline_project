import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Loads messages and categories dataset and merges them

    Arguments:
    messages_filepath -- filepath of disaster_messages.csv
    categories_filepath -- filepath of disaster_categories.csv

    Returns:
    df -- merged pandas dataframe                    
    """

    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, on='id')

    return(df)

def clean_data(df):
    """Cleans merged dataset created by load_data"""

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0, :]
    
    # extract a list of new column names for categories
    category_colnames = row.str[:-2]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert the category values to 0 1 integers
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Replace categories column in df with new category columns.
    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return(df)

def save_data(df, database_filename):
    """Saves cleaned dataframe as table called 'messages' in sql database"""

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False)    

def main():
    """ wrapper function that loads, cleans and saves data """

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

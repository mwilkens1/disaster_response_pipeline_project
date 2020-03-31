from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sqlalchemy import create_engine
import pandas as pd
import sys
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])


def load_data(database_filepath):
    """
    Loads data and creates dataframe
    
    Arguments:
    database_filepath -- filepath to the sql database

    Returns:
    X -- messages column
    Y -- dataframe of the categories
    category_names -- list of names of all the categories
    
    """
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table('messages', engine)
    
    X = df.message
    Y = df.iloc[:, 4:]
    category_names = Y.columns.to_list()

    return(X, Y, category_names)

def tokenize(message):
    """Tokenization function to process the text data"""

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%'\
        '[0-9a-fA-F][0-9a-fA-F]))+'

    # Removing any URLs
    detected_urls = re.findall(url_regex, message)

    for url in detected_urls:
        message = message.replace(url, "urlplaceholder")

    # Removing punctuation
    message = re.sub(r"[^a-zA-Z0-9]", " ", message)

    # Tokenizing the message
    tokens = word_tokenize(message)

    # Lemmatize, lowercase, strip and also removing stopwords
    clean_tokens = [WordNetLemmatizer().lemmatize(t).lower().strip()
                    for t in tokens if t not in stopwords.words("english")]

    return(clean_tokens)


def build_model():
    """
    Build machine leaning model

    Builds the machine learning pipeline by chaining CountVectorizer,
    TfidfTransformer and RandomForestClassifier. GridSearchCV is initiated 
    using a paramtergrid. 

    Returns an initated model, but does not fit the model

    """

    # Pipeline 
    pipeline = Pipeline([
        # counting word occurances with countvectorizer
        # calls tokenizer function
        ('vect', CountVectorizer(tokenizer=tokenize)),
        # followed by tfidftransfomer
        ('tfidf', TfidfTransformer()),
        # estimation with randomforestclassifier
        # uses multioutputclassifier to account for the multilabel nature
        ('clf', MultiOutputClassifier(RandomForestClassifier()))

    ])

    # Parameter grid
    # Note: this is a more focused version after a random grid search
    # See ML Pipeline Preparation for more
    #
    # I reduced the parameters here even further to reduce time
    # It would make more sense to me to not do a gridsearch in this script but
    # to fit the model with the tuned paramters (or not fit at all but use
    # the output from the notebook) but I suppose this is the assignment.
    parameters = {   
        'vect__ngram_range': [(1,1)],     
        'vect__max_df': (0.7, 0.75),
        'vect__max_features': (1000, 2000),
        'clf__estimator__n_estimators': [160],
        'clf__estimator__max_features': ["sqrt"], 
        'clf__estimator__min_samples_split': [5] 
    }

    # Gridsearch 
    cv = GridSearchCV(pipeline,
            param_grid = parameters,
            scoring = ['f1_micro','precision_micro','recall_micro'],
            refit ='f1_micro',
            verbose = 3,
            n_jobs = 1 # has to be set to 1 to avoid bug                 
    )  

    return(cv)

def evaluate_model(model, X_test, Y_test, category_names):
    """Shows the scores of the best fitted model as well as the paramters"""
    y_pred = model.predict(X_test)

    print(classification_report(Y_test, y_pred, target_names=category_names))
    print(model.best_params_)    

def save_model(model, model_filepath):
    """Pickles model"""
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

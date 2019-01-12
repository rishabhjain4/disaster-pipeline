import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import pickle
import nltk
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


def load_data(database_filepath):
    '''
    Load data frame from sql database.
    param: database_filepath
    return: X, Y, label_names
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('message_categories', con=engine)
    df = df.dropna()
    X = df.loc[:, 'message']
    Y = df.iloc[:, 4:]
    categories = list(Y)

    return X.values, Y.values, categories


def tokenize(text):
    '''
    param: text
    return: list of tokens
    '''

    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    stemmed = [WordNetLemmatizer().lemmatize(w) for w in words]

    return stemmed


def build_model():
    """
    Build Machine Learning Model
    Returns (model): Pipeline and gridsearch model
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])

    parameters = {'clf__estimator__min_samples_split': [2, 4, 6],
                  'clf__estimator__max_depth': [2, 4]}

    #parameters = {'clf__estimator__min_samples_split':[2]}
    model = GridSearchCV(pipeline, parameters)

    return(model)


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model performance and display aacuracy.
    param: model
    param: X_test
    param: Y_test
    param: category_names
    return: None
    '''

    predictions = model.predict(X_test)

    # Since there are 36 categories, we'll just loop over them to calculate the accuracy of each category.
    print("Accuracy scores for each category\n")
    print("*-" * 30)

    for i in range(36):
        print("Category:", category_names[i], "\n", classification_report(
            Y_test[:, i], predictions[:, i]))


def save_model(model, model_filepath):
    '''
    Save model in pickle format

    param: model
    param: model_filepath

    return: None
    '''
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''

    Load Data from Messages adn Categories file.
    Merge both into one dataframe.

    param: messages_filepath
    param: categories_filepath
    return: df 

    '''

    # load messages and categories data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge dataset
    df = pd.merge(messages, categories, on="id")

    return df


def clean_data(df):
    '''
    Takes dataframe and splits categories and split them into
    binary 0 or 1. Also dropping the duplicates.

    param:df
    return: df (Cleaned Dataframe)
    '''

    # split categories columns to new dataframe

    categories = df.categories.str.split(';', expand=True)

    # name the columns using the first row in categories dataframe

    row = categories.iloc[0, :].values
    category_col_names = list(map(lambda x: x[:-2], row))
    categories.columns = category_col_names

    # converte all the rows value to last character and convert it to numeric
    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x[-1]))

    # drop the original categories column from df

    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    '''
    Load dataframe to a SQLite database
    param:df Structured and cleaned dataframe
    param: database_filename
    '''

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('message_categories', engine, index=False)


def main():
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
# Disaster Response Pipeline Project

In this Project, data set containing real messages which were sent during disaster events is used to build a model to categorize events so that one can send the messages to an appropriate disaster relief agency.

Dataset used: disaster data from [Figure Eight](https://www.figure-eight.com/)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Important Files:
- `data/process_data.py`: The ETL pipeline used to process data in preparation.
- `models/train_classifier.py`: The Machine Learning pipeline used to fit, tune, evaluate, and export the model to a Python pickle.
- `app/templates/*.html`: HTML templates for the web app.
- `run.py`: Start the Python server for the web app and prepare visualizations.

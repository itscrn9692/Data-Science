import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the model
model = joblib.load('model.joblib')

# Define the Streamlit app
def app():
    # Load the dataset
    data = pd.read_csv('mushrooms.csv')

    # Add a title to the app
    st.title("Mushroom Edibility Predictor")

    # Create a list of all the columns in the dataset except the target column
    columns = data.columns.tolist()
    columns.remove("class")

    # Create a dictionary to store the user's selected values for each column
    user_inputs = {}

    # Loop through each column and add a dropdown menu for the user to select a value
    for column in columns:
        # Get the unique values for the column
        values = data[column].unique()

        # Sort the values in alphabetical order
        values = sorted(values)

        # Add a dropdown menu for the user to select a value
        user_inputs[column] = st.selectbox(f"Select a value for {column}", values)

    # Create a DataFrame with the user's selected values
    new_data = pd.DataFrame(user_inputs, index=[0])

    # Apply label encoding to the new data
    new_data = new_data.apply(LabelEncoder().fit_transform)

    # Make predictions on the new data
    prediction = model.predict(new_data)

    # Create a dictionary to map the label encoded values back to their corresponding labels
    label_map = {0: 'edible', 1: 'poisonous'}

    # Display the prediction to the user
    st.write(f"The mushroom is {label_map[prediction[0]]}")

    # Add a submit button
    if st.button("Submit"):
        st.write("Thank you for using the Mushroom Classifier!")

if __name__ == '__main__':
    app()

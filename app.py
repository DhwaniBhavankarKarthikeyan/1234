import streamlit as st
import numpy as np
import pandas as pd
#import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv("Finaldf-2.csv")
le_gender = LabelEncoder()
df['gender_encoded'] = le_gender.fit_transform(df['gender'])

# Preprocess your data and train models as needed

# Set Streamlit page title and icon
st.set_page_config(page_title="Fraud Detection App", page_icon="✅")

def knn_page(df):
    st.title("K-Nearest Neighbors (KNN) Page")
    
    # Sidebar options for KNN parameters
    st.sidebar.header("KNN Parameters")
    k_value = st.sidebar.slider("Select the number of neighbors (k)", 1, 20, 5)
    
    # Display the dataset and KNN results
    st.write("Your dataset:")
    st.write(df)  # You may want to display a subset of your data here
    
    # Split the data into features (X) and labels (y)
    X = df[['amt', 'lat', 'long', 'gender_encoded']]  # Include 'gender_encoded' as a feature
    y = df['is_fraud']
    
    # Split the data into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a KNN classifier and fit it to the training data
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = knn.predict(X_test)
    
    # Calculate and display the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy of the KNN model: {accuracy:.2f}")
    
    # Input fields for user to make predictions
    st.header("Make a KNN Prediction")
    amt = st.number_input("Transaction Amount")
    lat = st.number_input("Latitude")
    long = st.number_input("Longitude")
    gender = st.selectbox("Gender", df['gender'].unique())  # Allow all unique gender labels
    
    # Handle the case of previously unseen labels
    if gender not in df['gender'].unique():
        gender_encoded = -1  # Encode unseen labels as -1 (or choose another default value)
    else:
        gender_encoded = le_gender.transform([gender])[0]
    
    # Predict using the user's input
    input_data = [amt, lat, long, gender_encoded]  # Create input data with 'gender_encoded'
    prediction = knn.predict([input_data])
    
    # Display the prediction
    if prediction[0] == 0:
        st.write("Prediction: Not Fraudulent")
    else:
        st.write("Prediction: Fraudulent")

def nb_page(df):
    st.title("Naive Bayes Page")
    
    # Sidebar options for Naive Bayes parameters (if any)
    # Example: smoothing parameter
    # alpha = st.sidebar.slider("Smoothing (alpha)", 0.0, 1.0, 0.1)
    
    # Display the dataset and Naive Bayes results
    st.write("Your dataset:")
    st.write(df)  # You may want to display a subset of your data here
    
    # Split the data into features (X) and labels (y)
    X = df[['amt', 'lat', 'long', 'gender_encoded']]  # Include 'gender_encoded' as a feature
    y = df['is_fraud']
    
    # Split the data into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a Naive Bayes classifier and fit it to the training data (Gaussian Naive Bayes)
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = nb.predict(X_test)
    
    # Calculate and display the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy of the Naive Bayes model: {accuracy:.2f}")
    
    # Input fields for user to make predictions
    st.header("Make a Naive Bayes Prediction")
    amt = st.number_input("Transaction Amount")
    lat = st.number_input("Latitude")
    long = st.number_input("Longitude")
    gender = st.selectbox("Gender", df['gender'].unique())  # Allow all unique gender labels
    
    # Handle the case of previously unseen labels
    if gender not in df['gender'].unique():
        gender_encoded = -1  # Encode unseen labels as -1 (or choose another default value)
    else:
        gender_encoded = le_gender.transform([gender])[0]
    
    # Predict using the user's input
    input_data = [amt, lat, long, gender_encoded]  # Create input data with 'gender_encoded'
    prediction = nb.predict([input_data])
    
    # Display the prediction
    if prediction[0] == 0:
        st.write("Prediction: Not Fraudulent")
    else:
        st.write("Prediction: Fraudulent")

def logistic_page(df):
    st.title("Logistic Regression Page")
    
    # Sidebar options for Logistic Regression parameters (if any)
    # Example: regularization parameter (C)
    # C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
    
    # Display the dataset and Logistic Regression results
    st.write("Your dataset:")
    st.write(df)  # You may want to display a subset of your data here
    
    # Split the data into features (X) and labels (y)
    X = df[['amt', 'lat', 'long', 'gender_encoded']]  # Include 'gender_encoded' as a feature
    y = df['is_fraud']
    
    # Split the data into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a Logistic Regression classifier and fit it to the training data
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = logistic_model.predict(X_test)
    
    # Calculate and display the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy of the Logistic Regression model: {accuracy:.2f}")
    
    # Input fields for user to make predictions
    st.header("Make a Logistic Regression Prediction")
    amt = st.number_input("Transaction Amount")
    lat = st.number_input("Latitude")
    long = st.number_input("Longitude")
    gender = st.selectbox("Gender", df['gender'].unique())  # Allow all unique gender labels
    
    # Handle the case of previously unseen labels
    if gender not in df['gender'].unique():
        gender_encoded = -1  # Encode unseen labels as -1 (or choose another default value)
    else:
        gender_encoded = le_gender.transform([gender])[0]
    
    # Predict using the user's input
    input_data = [amt, lat, long, gender_encoded]  # Create input data with 'gender_encoded'
    prediction = logistic_model.predict([input_data])
    
    # Display the prediction
    if prediction[0] == 0:
        st.write("Prediction: Not Fraudulent")
    else:
        st.write("Prediction: Fraudulent")


def rf_page(df):
    st.title("Random Forest Classifier Page")
    
    # Sidebar options for Random Forest parameters (if any)
    # Example: number of trees (n_estimators)
    # n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 100, 50)
    
    # Display the dataset and Random Forest results
    st.write("Your dataset:")
    st.write(df)  # You may want to display a subset of your data here
    
    # Split the data into features (X) and labels (y)
    X = df[['amt', 'lat', 'long', 'gender_encoded']]  # Include 'gender_encoded' as a feature
    y = df['is_fraud']
    
    # Split the data into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a Random Forest classifier and fit it to the training data
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)
    
    # Calculate and display the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy of the Random Forest model: {accuracy:.2f}")
    
    # Input fields for user to make predictions
    st.header("Make a Random Forest Prediction")
    amt = st.number_input("Transaction Amount")
    lat = st.number_input("Latitude")
    long = st.number_input("Longitude")
    gender = st.selectbox("Gender", df['gender'].unique())  # Allow all unique gender labels
    
    # Handle the case of previously unseen labels
    if gender not in df['gender'].unique():
        gender_encoded = -1  # Encode unseen labels as -1 (or choose another default value)
    else:
        gender_encoded = le_gender.transform([gender])[0]
    
    # Predict using the user's input
    input_data = [amt, lat, long, gender_encoded]  # Create input data with 'gender_encoded'
    prediction = rf_model.predict([input_data])
    
    # Display the prediction
    if prediction[0] == 0:
        st.write("Prediction: Not Fraudulent")
    else:
        st.write("Prediction: Fraudulent")


def eda_page():
    st.title("Exploratory Data Analysis (EDA) Page")

    # Display images using a loop
    image_files = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg", "image6.jpg", "image7.jpg", "image8.jpg", "image9.jpg"]
    
    for image_file in image_files:
        st.image(image_file, caption='Image Caption', use_column_width=True)


# Create a sidebar navigation menu
st.sidebar.title("Navigation")
selected_page = st.sidebar.selectbox(
    "Choose a page:",
    ["EDA", "KNN", "Naive Bayes", "Logistic Regression", "Random Forest Classifier"]
)

# Display the selected page
if selected_page == "EDA":
    eda_page(df)
elif selected_page == "KNN":
    knn_page(df)
elif selected_page == "Naive Bayes":
    nb_page(df)
elif selected_page == "Logistic Regression":
    logistic_page(df)
elif selected_page == "Random Forest Classifier":
    rf_page(df)

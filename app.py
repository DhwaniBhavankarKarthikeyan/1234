import streamlit as st
import numpy as np
import pandas as pd
#import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
#from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv("Finaldf-2.csv")

# Preprocess your data and train models as needed

# Set Streamlit page title and icon
st.set_page_config(page_title="Fraud Detection App", page_icon="✅")

# Define functions for each page
def knn_page(df):
    st.title("K-Nearest Neighbors (KNN) Page")
    
    # Sidebar options for KNN parameters
    st.sidebar.header("KNN Parameters")
    k_value = st.sidebar.slider("Select the number of neighbors (k)", 1, 20, 5)
    
    # Display the dataset and KNN results
    st.write("Your dataset:")
    st.write(df)  # You may want to display a subset of your data here
    
    # Label encode non-numeric features (job and city)
    le_job = LabelEncoder()
    le_city = LabelEncoder()
    
    # Fit label encoders only on available data
    le_job.fit(df['job'].values)
    le_city.fit(df['city'].values)
    
    # Transform data, handling unseen labels
    job_encoded = le_job.transform([job])[0] if job in le_job.classes_ else -1
    city_encoded = le_city.transform([city])[0] if city in le_city.classes_ else -1
    
    # Split the data into features (X) and labels (y)
    X = df[['amt', 'lat', 'long', 'job_encoded', 'city_encoded']]  # Adjust features as needed
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
    job = st.text_input("Job")
    city = st.text_input("City")
    
    # Predict using the user's input
    prediction = knn.predict([[amt, lat, long, job_encoded, city_encoded]])
    
    # Display the prediction
    if prediction[0] == 0:
        st.write("Prediction: Not Fraudulent")
    else:
        st.write("Prediction: Fraudulent")

def nb_page():
    st.title("Naive Bayes Page")
    # Your Naive Bayes model code

def logistic_page():
    st.title("Logistic Regression Page")
    # Your Logistic Regression model code

def rf_page():
    st.title("Random Forest Classifier Page")
    # Your Random Forest model code

def eda_page():
    st.title("Exploratory Data Analysis (EDA) Page")
    # Your EDA code, e.g., use Plotly or other visualization libraries

# Create a sidebar navigation menu
st.sidebar.title("Navigation")
selected_page = st.sidebar.selectbox(
    "Choose a page:",
    ["EDA", "KNN", "Naive Bayes", "Logistic Regression", "Random Forest Classifier"]
)

# Display the selected page
if selected_page == "EDA":
    eda_page()
elif selected_page == "KNN":
    knn_page(df)
elif selected_page == "Naive Bayes":
    nb_page()
elif selected_page == "Logistic Regression":
    logistic_page()
elif selected_page == "Random Forest Classifier":
    rf_page()

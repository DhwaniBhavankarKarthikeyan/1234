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

    # Handle missing or empty values in categorical features
    df['city'] = df['city'].fillna('Unknown')  # Replace missing values with a label
    df['job'] = df['job'].fillna('Unknown')  # Replace missing values with a label

    # Perform one-hot encoding for categorical features
    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(df[['city', 'job']])

    # Combine encoded features with numeric features
    X = np.column_stack((encoded_features.toarray(), df[['amt', 'lat', 'long']])).astype(float)
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
    
    # Input fields for the non-numeric features
    city = st.text_input("City")
    job = st.text_input("Job")

    # One-hot encode the user inputs
    user_input = encoder.transform(np.array([[city, job]]).reshape(1, -1))
    
    # Combine user inputs with numeric features
    user_features = np.column_stack((user_input.toarray(), np.array([[amt, lat, long]])).astype(float)
    
    # Predict using the user's input
    prediction = knn.predict(user_features)
    
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

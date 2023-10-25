import streamlit as st
import numpy as np
import pandas as pd
#import plotly.express as px
#from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier 
#from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score

# Load your dataset
#df = pd.read_csv("Finaldf-2.csv")

# Preprocess your data and train models as needed

# Set Streamlit page title and icon
st.set_page_config(page_title="Fraud Detection App", page_icon="✅")

# Define functions for each page
def knn_page():
    st.title("K-Nearest Neighbors (KNN) Page")
    # Your KNN model code

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
    knn_page()
elif selected_page == "Naive Bayes":
    nb_page()
elif selected_page == "Logistic Regression":
    logistic_page()
elif selected_page == "Random Forest Classifier":
    rf_page()

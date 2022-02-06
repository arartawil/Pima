import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# Pima Indian Diabetes Prediction App
**Diabetes ** Type 2!
""")

st.sidebar.header('User Input Features')

#st.sidebar.markdown("""
#[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
#""")

# Collects user input features into dataframe
#uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
uploaded_file=None;
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():

        Pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 10)
        Glucose = st.sidebar.slider('Glucose', 0, 199, 100)
        BloodPressure = st.sidebar.slider('Blood Pressure', 0, 122, 50)
        SkinThickness = st.sidebar.slider('Skin Thickness', 0, 99, 10)
        Insulin = st.sidebar.slider('Insulin', 0, 846, 100)
        BMI = st.sidebar.slider('Body Mass Index', 10.0 ,67.1 ,17.2)
        DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree (Family history)', 0.08, 2.42, 1.0)
        Age = st.sidebar.slider('Age', 10, 80, 60)
        data = {
                'Pregnancies': Pregnancies,
                'Glucose': Glucose,
                'BloodPressure': BloodPressure,
                'SkinThickness': SkinThickness,
                'Insulin': Insulin,
                'BMI': BMI,
                'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
                'Age': Age,
        }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
#Pima_ = pd.read_csv('Pima Indians Diabetes Database.csv')
#Pima = Pima_.drop(columns=['Outcome'])
#df = pd.concat([input_df,Pima],axis=0)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_df)

# Reads in saved classification model
load_clf = pickle.load(open('Pima_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)


st.subheader('Prediction')
penguins_species = np.array(['Negative','Positive'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

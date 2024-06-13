import streamlit as st
import pickle
import numpy as np
import pandas
lr1 = pickle.load(open())
dt1 = pickle.load(open())
rf1 = pickle.load(open())
st.title('Insurance Charge Prediction App')
st.header('Fill the details to generate the Predicted Insurance Charge')
options = st.sidebar.selectbox('select ML Model', ['Lin_Reg', 'Decision_Tree', 'Random_Forest'])
age = st.slider('Age', 18, 64)
sex = st.selectbox('Sex', ['Male', 'Female'])
bmi = st.slider('BMI', 15, 53)
children = st.selectbox('Children', [0, 1, 2, 3, 4, 5])
smoker = st.selectbox('Smoker', ['Yes', 'No'])
region = st.selectbox('Region', ['NWest', 'SEast', 'SWest', 'NEast'])
if st.button('Predict'):
    if sex == 'Male':
        sex = 1
    else:
        sex = 0
    if smoker == 'Yes':
        smoker = 1
    else:
        smoker = 0
    if region == 'NWest':
        region = 1
    elif region == 'NEast':
        region = 0
    elif region == 'SEast':
        region = 2
    else:
        region = 3
    test = np.array([age, sex, bmi, children, smoker, region])
    test = test.reshape(1, 6)
    if options == 'Lin_Reg':
        st.success(lr1.predict(test)[0])
    elif options == 'Decision_Tree':
        st.success(dt1.predict(test)[0])
    else:
        st.success(rf1.predict(test)[0])
import streamlit as st
import numpy as np
from joblib import load

# Function to load models safely
def load_model(filename):
    try:
        return load(filename)
    except FileNotFoundError:
        st.error(f"Model file '{filename}' not found. Ensure it is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model '{filename}': {e}")
        return None

# Load models
lr1 = load_model("lr1.joblib")
dt1 = load_model("dt1.joblib")
rf1 = load_model("rf1.joblib")

# App title and description
st.title('Insurance Charge Prediction App')
st.header('Fill in the details to generate the predicted insurance charge')

# Sidebar for model selection
options = st.sidebar.selectbox('Select ML Model', ['Lin_Reg', 'Decision_Tree', 'Random_Forest'])

# Input fields
age = st.slider('Age', 18, 64, step=1)
sex = st.selectbox('Sex', ['Male', 'Female'])
bmi = st.slider('BMI', 15.0, 53.0, step=0.1)
children = st.selectbox('Children', [0, 1, 2, 3, 4, 5])
smoker = st.selectbox('Smoker', ['Yes', 'No'])
region = st.selectbox('Region', ['NWest', 'SEast', 'SWest', 'NEast'])

# Prediction button
if st.button('Predict'):
    # Encode categorical variables
    sex = 1 if sex == 'Male' else 0
    smoker = 1 if smoker == 'Yes' else 0
    region_mapping = {'NWest': 1, 'NEast': 0, 'SEast': 2, 'SWest': 3}
    region = region_mapping[region]

    # Prepare input array
    test_input = np.array([age, sex, bmi, children, smoker, region]).reshape(1, -1)

    # Check selected model and make predictions
    if options == 'Lin_Reg' and lr1:
        prediction = lr1.predict(test_input)[0]
        st.success(f"Predicted Insurance Charge (Lin_Reg): ${prediction:.2f}")
    elif options == 'Decision_Tree' and dt1:
        prediction = dt1.predict(test_input)[0]
        st.success(f"Predicted Insurance Charge (Decision_Tree): ${prediction:.2f}")
    elif options == 'Random_Forest' and rf1:
        prediction = rf1.predict(test_input)[0]
        st.success(f"Predicted Insurance Charge (Random_Forest): ${prediction:.2f}")
    else:
        st.error("Selected model is not loaded or invalid.")

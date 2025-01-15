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
models = {
    "Lin_Reg": load_model("lr1.joblib"),
    "Decision_Tree": load_model("dt1.joblib"),
    "Random_Forest": load_model("rf1.joblib"),
}

# App title and description
st.title('Insurance Charge Prediction App')
st.header('Fill in the details to generate the predicted insurance charge')

# Sidebar for model selection
selected_model = st.sidebar.selectbox('Select ML Model', list(models.keys()))

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

    # Get the selected model
    model = models[selected_model]

    if model:
        try:
            # Make prediction
            prediction = model.predict(test_input)[0]
            st.success(f"Predicted Insurance Charge ({selected_model}): ${prediction:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error(f"The selected model '{selected_model}' could not be loaded.")

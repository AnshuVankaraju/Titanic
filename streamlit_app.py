import joblib
import numpy as np
import streamlit as st

model = joblib.load('titanic_model.pkl')
st.title("TITANIC SURVIVAL PREDICTION APP")

#Input : PClass
pclass = st.selectbox('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)', [1, 2, 3])

# Input: Gender
gender = st.selectbox('Gender (Male = 1, Female = 0)', ['Male', 'Female'])
is_male = 1 if gender == 'Male' else 0

# Input: Age
age = st.slider('Select Age', min_value=0, max_value=100, value=25)

# Input: Siblings/Spouses
sib_spo = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)

# Input: Parents/Children
par_chil = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, value=0)

# Input: Fare
fare = st.number_input("Ticket Fare", min_value=0.0, max_value=600.0, value=50.0, step=5.0)

# Input: Embarked
embarked = st.selectbox('Port of Embarkation', ['Cherbourg (C)', 'Queenstown (Q)', 'Southampton (S)'])
if 'Cherbourg' in embarked:
    embarked_code = 0
elif 'Queenstown' in embarked:
    embarked_code = 1
else:
    embarked_code = 2

# Predict button
if st.button("Predict"):
    input_data = np.array([[pclass, is_male, age, sib_spo, par_chil, fare, embarked_code]])
    
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")

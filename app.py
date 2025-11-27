import streamlit as st
import pandas as pd
from pickle import load
import numpy as np

st.title("Classification Model")




model = load(open("model.pkl", "rb"))


st.title("Titanic Survival Prediction")

Pclass = st.selectbox("Passenger Class", [1,2,3])
Sex = st.selectbox("Sex", ["male","female"])
Age = st.slider("Age", 1, 80)
Fare = st.number_input("Fare")
SibSp = st.number_input("Siblings/Spouses", 0, 10)
Parch = st.number_input("Parents/Children", 0, 10)
Embarked = st.selectbox("Embarked", ['S','C','Q'])

# Encode
Sex = 1 if Sex=="female" else 0
Embarked = {'S': 2, 'C': 0, 'Q': 1}[Embarked]

input_data = np.array([[Pclass,Sex,Age,SibSp,Parch,Fare,Embarked]])

pred = model.predict(input_data)[0]

if pred == 1:
    st.success("🎉 Passenger Survived")
else:
    st.error("❌ Passenger Did Not Survive")


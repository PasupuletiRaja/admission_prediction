import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('ad_model.pkl')
scaler = joblib.load('ad_scaler.pkl')


#scaler = joblib.load('scaler1.pkl')

def predict_probability(gre, toefl, uni_rating, sop, lor, cgpa, research):
    input_data = [[gre, toefl, uni_rating, sop, lor, cgpa, research]]
    input_data = scaler.transform(input_data)
    admission_probability = model.predict_proba(input_data)[:,1]
    return admission_probability[0]

def main():
    st.title("Graduate Admissions Predictor")
    st.write("Please enter your information below:")

    gre = st.number_input("GRE Score (<340)", min_value=0, max_value=340)
    toefl = st.number_input("TOEFL Score (<120)", min_value=0, max_value=120)
    uni_rating = st.selectbox("University Rating (1-5)", options=[1, 2, 3, 4, 5])
    sop = st.slider("SOP Rating (1-5)", min_value=1.0, max_value=5.0, step=0.5, format="%.1f")
    lor = st.slider("LOR Rating (1-5)", min_value=1.0, max_value=5.0, step=0.5, format="%.1f")
    cgpa = st.number_input("CGPA (1-10)", min_value=0.0, max_value=10.0, step=0.1)
    research = st.selectbox("Research (1 if yes, 0 if no)", options=[0, 1])

    if st.button("Submit"):
        st.write("GRE Score:", gre)
        st.write("TOEFL Score:", toefl)
        st.write("University Rating:", uni_rating)
        st.write("SOP Rating:", sop)
        st.write("LOR Rating:", lor)
        st.write("CGPA:", cgpa)
        st.write("Research:", research)

        final_features = pd.DataFrame([[gre, toefl, uni_rating, sop, lor, cgpa, research]])
        output = predict_probability(gre, toefl, uni_rating, sop, lor, cgpa, research)
        st.write("Admission chances are {:.2f}%".format(output*100))


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
Spyder Editor/VSCode
This is a temporary script file.
"""
import pickle
from matplotlib.ft2font import BOLD
import streamlit as st
from streamlit_option_menu import option_menu


diabetes_model = pickle.load(open("Saved Models/diabetes_model.sav","rb"))

heart_model = pickle.load(open("Saved Models/heart_model.sav","rb"))

parkinsons_model = pickle.load(open("Saved Models/parkinsons.sav","rb"))

breast_cancer_model = pickle.load(open("Saved Models/breastCancer.sav","rb"))

with st.sidebar:
    
    selected = option_menu("Multiple Disease Prediction System",
                           
                            ["Diabetes Prediction",
                             "Heart Disease Prediction",
                             "Parkinson's Disease Prediction",
                             "Breast Cancer Prediction"],
                            
                            icons = ["activity","heart","person-fill","clipboard2-pulse-fill"],
                            
                            default_index = 0)
    

if selected == "Diabetes Prediction":
    
    st.title("Diabetes Prediction using Machine Learning")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        Insulin = st.text_input('Insulin Level')
    
    with col1:
        BMI = st.text_input('BMI')
    
    with col2:
        Diabetes_Pedigree_Function = st.text_input('Diabetes Pedigree Function')
    
    with col3:
        Age = st.text_input('Age')
   

    diab_diagnosis=""
    
    if st.button("Diabetes Test Prediction"):
        diabetes_prediction = diabetes_model.predict([[Pregnancies,Glucose,Insulin,BMI,Diabetes_Pedigree_Function,Age]])
        
        if diabetes_prediction[0]==1:
            diab_diagnosis="Prediction Result: Diabetes Positive"
            
        else:
            diab_diagnosis="Prediction Result: Diabetes Negative"
    
    st.success(diab_diagnosis)
  



              
if selected == "Heart Disease Prediction":
    
    st.title("Heart Disease Prediction using Machine Learning")

    with st.container():
        col1,col2,col3 = st.columns(3)

        with col1:
            st.markdown("**A. Chest Pain Types:**  \n0-> typical angina  \n1-> atypical angina  \n2-> non-anginal pain  \n3-> asymptomatic  \n")
        
        with col2:
            st.markdown("**B. Fasting Blood Sugar:**  \n1-> Greater than 120mg/dl  \n0-> Lesser than or equal to 120mg/dl")
        
        with col3:
            st.markdown("**C. Resting ECG:** \n0-> Normal  \n1-> Having ST-T wave abnormality  \n2-> showing probable or definite left ventricular hypertrophy by Estes' criteria\n")

        with col1:
            st.markdown("**D. Excercise Induced Angina:**  \n0-> yes  \n1-> no\n")
        
        with col2:
            st.markdown("**E. Slope:**  \n0-> Upsloping  \n1-> Flat \n 2->Downsloping\n")
        
        
    st.markdown("***")
    with st.container():
    
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            age = st.text_input('Age')
            
        with col2:
            cp = st.selectbox('Chest Pain',[0,1,2,3])
            
        with col3:
            trestbps = st.text_input('Resting Blood Pressure')
            
        with col4:
            chol = st.text_input('Serum Cholestoral in mg/dl')
            
        with col1:
            fbs = st.selectbox('Fasting Blood Sugar',[0,1],index=0)
            
        with col2:
            restecg = st.selectbox('Resting ECG',[0,1,2],index=0)
            
        with col3:
            thalach = st.text_input('Maximum Heart Rate')
        
        with col4:
            exang = st.text_input('Excercise Induced Angina')

        with col1:
            oldpeak = st.text_input('ST depression Induced')
            
        with col2:
            slope = st.selectbox('Slope ',[0,1,2])
            
        with col3:
            ca = st.selectbox('Blood Vessels coloured by flourosopy',[0,1,2,3])
            
        
        heart_diagnosis = ""

        if st.button('Heart Disease Test Prediction'):
            heart_prediction = heart_model.predict([[age, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca]])                          
            
            if (heart_prediction[0] == 1):
                heart_diagnosis = "Prediction Result: Heart Disease Positive"
            else:
                heart_diagnosis = "Prediction Result: Heart Disease Negative"
            
        st.success(heart_diagnosis)




if selected == "Parkinson's Disease Prediction":
    
    st.title("Parkinson's Disease Prediction using Machine Learning")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col3:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col4:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col5:
        RAP = st.text_input('MDVP:RAP')
        
    with col1:
        PPQ = st.text_input('MDVP:PPQ')

    with col2:
        shimmer = st.text_input('MDVP:Shimmer')

    with col3:
        shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col4:
        APQ = st.text_input('MDVP:APQ')    
    
    with col5:
        NHR = st.text_input("NHR")

    with col1:
        DFA = st.text_input("DFA")

    with col2:
        spread1 = st.text_input("spread1")

    with col3:
        PPE = st.text_input("PPE")

    with col4:
        spread2 = st.text_input("spread2")

    with col5:
        HNR = st.text_input("HNR")
    
    with col1:
        RPDE = st.text_input("RPDE")
        
    parkinsons_diagnosis = ''
        
    if st.button("Parkinson's Disease Test Prediction"):
        parkinsons_prediction = parkinsons_model.predict([[fo, Jitter_percent, fhi,  Jitter_Abs, RAP, PPQ,shimmer,shimmer_dB,APQ,NHR,DFA,spread1,PPE,spread2,HNR,RPDE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "Prediction Result: Parkinson's Positive"
        else:
          parkinsons_diagnosis = "Prediction Result: Parkinson's Negative"
        
    st.success(parkinsons_diagnosis)
        
        
if selected == "Breast Cancer Prediction":
    
    st.title("Breast Cancer Prediction using Machine Learning")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        radius_mean = st.text_input('Radius Mean')
        
    with col2:
        texture_mean = st.text_input('Texture Mean')
        
    with col3:
        perimeter_mean = st.text_input('Perimeter Mean')
        
    with col4:
        area_mean = st.text_input('Area Mean')
        
    with col5:
        smoothness_mean = st.text_input('Smoothness Mean')

    with col1:
        compactness_mean = st.text_input("Compactness Mean")
    
    with col2:
        concavity_mean = st.text_input('Concavity Mean')
        
    with col3:
        concave_points_mean = st.text_input('Concave Pts Mean')
        
    with col4:
        symmetry_mean = st.text_input('Symmetry Mean')
        
    with col5:
        radius_se = st.text_input('Radius SE')
    
    with col1:
        perimeter_se = st.text_input('Perimeter SE')

    with col2:
        area_se = st.text_input('Area SE')

    with col3:
        concave_points_se = st.text_input('Concave Pts SE')
    
    with col4:
        radius_worst = st.text_input("Radius Worst")
    
    with col5:
        texture_worst = st.text_input('Texture Worst')
        
    with col1:
        perimeter_worst = st.text_input('Perimeter Worst')
        
    with col2:
        area_worst = st.text_input('Area Worst')
        
    with col3:
        smoothness_worst = st.text_input('Smoothness Worst')

    with col4:
        compactness_worst = st.text_input("Compactness Worst")
    
    with col5:
        concavity_worst = st.text_input('Concavity Worst')
        
    with col1:
        concave_points_worst = st.text_input('Concave Pts Worst')
        
    with col2:
        symmetry_worst = st.text_input('Symmetry Worst')
        
    with col3:
        fractal_dimension_worst = st.text_input('Fractional Dimension Worst')

    breast_cancer_diagnosis = ''
        
    if st.button("Breast Cancer Test Prediction"):
        breast_cancer_prediction = breast_cancer_model.predict([[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,radius_se,perimeter_se,area_se,concave_points_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]])                          
        
        if (breast_cancer_prediction[0] == 1):
          breast_cancer_diagnosis = "Prediction Result: Breast Cancer Positive"
        else:
          breast_cancer_diagnosis = "Prediction Result: Breast Cancer Negative"
        
    st.success(breast_cancer_diagnosis)

# ==========================
# Gradio Diabetes Predictor
# ==========================
import gradio as gr
import pandas as pd
import pickle
import numpy as np

#load model
with open("diabeties_pred.pkl", "rb") as f: \
    model = pickle.load(f)

# main function
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness,
                     Insulin, BMI, DiabetesPedigreeFunction, Age):
    
    # Pack inputs into a DataFrame
    input_df = pd.DataFrame([[
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    ]],
    columns=['Pregnancies','Glucose','BloodPressure','SkinThickness',
             'Insulin','BMI','DiabetesPedigreeFunction','Age'])
    

    prediction = model.predict(input_df)[0]
    
    # Return formatted result
    return "Diabetic" if prediction == 1 else "Non-Diabetic"

#main app interface 
inputs = [
    gr.Number(label="Pregnancies (Number of Babies)"), 
    gr.Slider(70, 200, step=1, label="Glucose"),  
    gr.Slider(50, 100, step=1, label="BloodPressure"),
    gr.Slider(10, 50, step=1, label="SkinThickness"),
    gr.Slider(15, 200, step=1, label="Insulin"),
    gr.Slider(15, 50, step=0.1, label="BMI"),
    gr.Slider(0.0, 2.5, step=0.01, label="DiabetesPedigreeFunction"),
    gr.Slider(21, 70, step=1, label="Age")
]

app = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs="text",
    title="Diabetes Predictor"
)

# Launch

app.launch(share=True)
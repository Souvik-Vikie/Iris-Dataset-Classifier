# streamlit_app/main.py
import os
import json
import requests
import streamlit as st

# FastAPI endpoint (use env var in Render; fallback to your live backend)
API_URL = os.getenv("BACKEND_URL", "https://iris-souvik-fastapi.onrender.com/predict")

# Model options dictionary (match FastAPI routes exactly)
models = {
    "DecisionTree": "DecisionTree",
    "KNN": "KNN",
    "LogisticRegression": "LogisticRegression",
}

def main():
    st.title("Machine Learning Model Predictor")

    # Model selection dropdown
    selected_model = st.selectbox("Select a model", list(models.keys()))
    model_file = models[selected_model]

    # Feature inputs
    sepal_length = st.number_input("Sepal length", value=5.1, format="%.2f")
    sepal_width  = st.number_input("Sepal width",  value=3.5, format="%.2f")
    petal_length = st.number_input("Petal length", value=1.4, format="%.2f")
    petal_width  = st.number_input("Petal width",  value=0.2, format="%.2f")

    if st.button("Predict"):
        # Prepare features as the backend expects: {"data": [..]}
        feature_data = [sepal_length, sepal_width, petal_length, petal_width]
        payload = {"data": feature_data}

        try:
            resp = requests.post(f"{API_URL}/{model_file}", json=payload, timeout=30)
            if resp.ok:
                # Expecting {"model": "...", "prediction": <int>}
                result = resp.json()
                st.success(f"Model: {result.get('model', selected_model)} | Prediction: {result.get('prediction')}")
            else:
                # Show backend error body to help debugging
                st.error(f"Error {resp.status_code}: {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

if __name__ == "__main__":
    main()






# import streamlit as st
# import requests
# import json

# # FastAPI endpoint
# API_URL = "http://localhost:8000/predict"

# # Model options dictionary
# models = {
#     "DecisionTree": "DecisionTree",
#     "KNN": "KNN",
#     "LogisticRegression": "logistic_regression_model"
# }

# # Streamlit app
# def main():
#     st.title("Machine Learning Model Predictor")

#     # Model selection dropdown
#     selected_model = st.selectbox("Select a model", list(models.keys()))

#     # Get model file name based on selection
#     model_file = models[selected_model]

#     # Feature inputs
#     sepal_length = st.number_input("Sepal length")
#     sepal_width = st.number_input("Sepal width")
#     petal_length = st.number_input("Petal length")
#     petal_width = st.number_input("Petal width")

#     # Make prediction on button click
#     if st.button("Predict"):
#         # Prepare feature data as JSON payload
#         feature_data = {
#             "sepal_length": sepal_length,
#             "sepal_width": sepal_width,
#             "petal_length": petal_length,
#             "petal_width": petal_width
#         }
#         feature_data = [sepal_length, sepal_width, petal_length, petal_width]

#         # Call FastAPI endpoint and get prediction result
#         headers = {'Content-Type': 'application/json'}
#         response = requests.post(API_URL + f"/{model_file}", json={"data": feature_data})

#         # Display prediction result
#         st.write(f"Prediction: {response.json()}")
    
# if __name__ == "__main__":
#     main()



import streamlit as st
import numpy as np
import cv2
import pickle
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Load the pickled model
with open("xception-model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Function to preprocess the uploaded image
def preprocess_image(image):
    img_array = np.array(image)  # Convert PIL image to numpy array
    img_array = cv2.resize(img_array, (256, 256))  # Resize image to model input size without specifying the number of channels
    img_array = img_array.astype('float32') / 255.0  # Normalize pixel values
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension if it's missing
    # print(img_array.shape,"\n\n\n")
    return img_array

# Function to make predictions
def predict_image(image):
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    return prediction

# Streamlit application
def main():
    st.title("Image Classification")
    st.write("Xception model is used with 99% of accuracy on trainig data and 65% accuracy on validation data")
    st.sidebar.title("Options")

    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

        if st.button("Predict"):
            prediction = np.argmax(predict_image(image))
            prediction=le.inverse_transform([prediction])
            st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()

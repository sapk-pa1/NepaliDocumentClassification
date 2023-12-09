import streamlit as st
import numpy as np
from src.data_preprocessing.preprocess import process_img
from src.models.load_model import load_resnet50_doc_class
from PIL import Image
st.set_page_config(layout="wide")


# Streamlit app
# Centered and enlarged title using Markdown and HTML
st.markdown("""
    <style>
        .title {
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
            font-size: 48px;  # Adjust the font size here
            font-weight: bold;
        }
    </style>
    <div class="title">
        Image Classification App
    </div>
    """, unsafe_allow_html=True)
# Load the pre-trained model
model = load_resnet50_doc_class()

# Create two columns
col1,_, col2 = st.columns([12,13,8])

# Column 1 for input
with col1:
    

    st.header("Input")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

# Column 2 for result
with col2:
    st.header("Prediction")
    if uploaded_file is not None:
        # Create a placeholder for the "Classifying..." message
        classifying_placeholder = st.empty()

        # Display the "Classifying..." message in the placeholder
        classifying_placeholder.text("Classifying...")

        # Read the content of the uploaded file
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Process and classify the image
        processed_image = process_img(image_np)  # Assuming process_img takes a NumPy array
        predictions = model.predict(processed_image)
        class_names = ['Citizenship', 'License', 'Passport']
        predicted_class = class_names[np.argmax(predictions)]

        # Clear the "Classifying..." message from the placeholder
        classifying_placeholder.empty()

        # Display the classification result
        st.write(f"Document Type: {predicted_class}")
        st.write(f"Confidence: {np.max(predictions):.2f}")
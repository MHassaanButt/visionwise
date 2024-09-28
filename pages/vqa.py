import streamlit as st
from PIL import Image
from inference.inference_utils import run_inference


# Visual Question Answering Page
def vqa_page():
    st.title("Visual Question Answering")

    # Upload an image file
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Input question from user
        question = st.text_input("Ask a question about the image")

        # Submit button
        if st.button("Submit"):
            if question:
                # Define task
                task = "<VQA>"

                # Run inference and get the VQA result
                response = run_inference(image=image, task=task, text=question)

                # Display the VQA result professionally
                st.write("**Answer:**")
                st.markdown(
                    f'<div style="font-size:20px;color:blue;">{response["<VQA>"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Please enter a question.")

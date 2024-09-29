import streamlit as st
import supervision as sv
from PIL import Image
from inference.inference_utils import model_load, run_inference

# Visual Question Answering Page
def ocr_page():
    st.title("Optical Character Recognition")

    # Upload an image file
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Captioning options
        task_option = st.selectbox(
            "Select Task", ["Simple OCR", "OCR with Region"]
        )
        if task_option == "Simple OCR":
        # Submit button
            if st.button("Submit"):
                task = "<OCR>"
                response = run_inference(image=image, task=task)
                st.write("OCR Response:", response[task])
                
        elif task_option == "OCR with Region":
            if st.button("Submit"):
                task = "<OCR_WITH_REGION>"

                response = run_inference(image=image, task=task)
                detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)

                bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
                label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

                image = bounding_box_annotator.annotate(image, detections)
                image = label_annotator.annotate(image, detections)
                image.thumbnail((600, 600))
                st.image(
                    image, caption="OCR with Region Output", use_column_width=True
                )
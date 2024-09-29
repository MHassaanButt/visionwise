import streamlit as st
import supervision as sv
from PIL import Image
from inference.inference_utils import model_load, run_inference

# Visual Question Answering Page
def obj_detection_page():
    st.title("Object Detection")

    # Upload an image file
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Captioning options
        task_option = st.selectbox(
            "Select Task", ["Simple Object Detection", "Open Vocabulary Detection", "Region Proposal"]
        )
        if task_option == "Simple Object Detection":
        # Submit button
            if st.button("Submit"):

                text = "<OD>"
                task = "<OD>"

                DEVICE, model, processor = model_load()
                inputs = processor(text=text, images=image, return_tensors="pt").to(DEVICE)
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                response = processor.post_process_generation(generated_text, task=task, image_size=image.size)
                detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)

                bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
                label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

                image = bounding_box_annotator.annotate(image, detections)
                image = label_annotator.annotate(image, detections)
                image.thumbnail((600, 600))

                st.image(
                    image, caption="Object Detection Result", use_column_width=True
                )
        # else:
        elif task_option == "Open Vocabulary Detection":
            user_text = st.text_input("Enter text for phrase grounding", value="")

            if st.button("Submit"):

                task = "<OPEN_VOCABULARY_DETECTION>"
                
                response = run_inference(image=image, task=task, text=user_text)
                detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)

                bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
                label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

                image = bounding_box_annotator.annotate(image, detections)
                image = label_annotator.annotate(image, detections)
                image.thumbnail((600, 600))

                st.image(
                    image, caption="Open Vocabulary Detection", use_column_width=True
                )
            
        elif task_option == "Region Proposal":
               if st.button("Submit"):

                task = "<REGION_PROPOSAL>"
                
                response = run_inference(image=image, task=task)
                detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)

                bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
                label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

                image = bounding_box_annotator.annotate(image, detections)
                image = label_annotator.annotate(image, detections)
                image.thumbnail((600, 600))

                st.image(
                    image, caption="Region Proposal Output", use_column_width=True
                )
            
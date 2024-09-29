import streamlit as st
import supervision as sv
from PIL import Image
from inference.inference_utils import model_load

# Visual Question Answering Page
def obj_detection_page():
    st.title("Object Detection")

    # Upload an image file
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

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
        #     st.write("Please submit the image in correct format.")
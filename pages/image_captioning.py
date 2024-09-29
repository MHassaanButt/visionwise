import streamlit as st
from PIL import Image
from inference.inference_utils import run_inference
import supervision as sv


def image_captioning_page():
    st.title("Image Captioning")
    st.write("Upload an image and select a captioning task.")

    # Image Upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Captioning options
        task_option = st.selectbox(
            "Select Task", ["Simple Captioning", "Advanced Captioning"]
        )

        if task_option == "Simple Captioning":
            caption_types = {
                "Simple Caption": "<CAPTION>",
                "Detailed Caption": "<DETAILED_CAPTION>",
                "More Detailed Caption": "<MORE_DETAILED_CAPTION>",
            }

            simple_task = st.radio(
                "Select Captioning Type",
                list(caption_types.keys()),
            )

            # Simple captioning logic
            if st.button("Submit"):
                # st.write(f"Running {simple_task} task...")
                response = run_inference(image=image, task=caption_types[simple_task])
                st.write("Caption:", response[caption_types[simple_task]])

        elif task_option == "Advanced Captioning":
            advance_caption_types = {
                "Dense Region Caption" : "<DENSE_REGION_CAPTION>",
                "Grounded Phrase": "<CAPTION_TO_PHRASE_GROUNDING>",
                "Grounded Phrase with More Detailed Caption": "<DETAILED_CAPTION> + <CAPTION_TO_PHRASE_GROUNDING>",
            }
            advanced_task = st.radio(
                "Select Advanced Task", list(advance_caption_types.keys())
            )

            user_text = ""
            # Get the corresponding task from the selected advanced task
            selected_task = advance_caption_types[advanced_task]

            if selected_task == "<CAPTION_TO_PHRASE_GROUNDING>":
                user_text = st.text_input("Enter text for phrase grounding", value="")
            elif selected_task == "<DETAILED_CAPTION> + <CAPTION_TO_PHRASE_GROUNDING>":
                # Default to the previously generated detailed caption (if any)
                if "detailed_caption" not in st.session_state:
                    st.session_state.detailed_caption = ""

                # Show the existing detailed caption if available
                user_text = st.text_input(
                    "Enter text for phrase grounding",
                    value=st.session_state.detailed_caption,
                )

            if st.button("Submit"):
                # st.write(f"Running {selected_task} task...")
                if (
                    selected_task
                    == "<DENSE_REGION_CAPTION>"
                ):
                    # First run the detailed caption task
                    task = "<DENSE_REGION_CAPTION>"

                    response = run_inference(image=image, task=task)
                    detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)

                    bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
                    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

                    image = bounding_box_annotator.annotate(image, detections)
                    image = label_annotator.annotate(image, detections)
                    image.thumbnail((600, 600))

                    st.image(
                        image,
                        caption="Dense Region Caption Result",
                        use_column_width=True,
                    )

                elif selected_task == "<CAPTION_TO_PHRASE_GROUNDING>":
                    response = run_inference(
                        image=image, task=selected_task, text=user_text
                    )
                    # st.write("Grounded Phrase Response:", response)
                    # Processing detections
                    detections = sv.Detections.from_lmm(
                        sv.LMM.FLORENCE_2, response, resolution_wh=image.size
                    )

                    bounding_box_annotator = sv.BoundingBoxAnnotator(
                        color_lookup=sv.ColorLookup.INDEX
                    )
                    label_annotator = sv.LabelAnnotator(
                        color_lookup=sv.ColorLookup.INDEX
                    )

                    image = bounding_box_annotator.annotate(image, detections)
                    image = label_annotator.annotate(image, detections)
                    image.thumbnail((600, 600))
                    # image.save("output_verfication.png")
                    # Show outputs
                    # st.write("Grounded Phrase Response:", response[selected_task])
                    st.image(
                        image, caption="Grounded Phrase Result", use_column_width=True
                    )

                elif (
                    selected_task
                    == "<DETAILED_CAPTION> + <CAPTION_TO_PHRASE_GROUNDING>"
                ):
                    # First run the detailed caption task
                    detailed_caption_task = "<DETAILED_CAPTION>"
                    response = run_inference(image=image, task=detailed_caption_task)
                    caption_text = response[detailed_caption_task]

                    # Store the detailed caption in the session state
                    # st.session_state.detailed_caption = caption_text

                    # Then run the caption to phrase grounding task with the generated caption text
                    phrase_grounding_task = "<CAPTION_TO_PHRASE_GROUNDING>"
                    # user_text = st.text_input(
                    #     "Enter text for phrase grounding", value=caption_text
                    # )  # Use the caption as default text

                    # if st.button("Submit"):
                    response = run_inference(
                        image=image, task=phrase_grounding_task, text=caption_text
                    )

                    # Processing and displaying detections
                    detections = sv.Detections.from_lmm(
                        sv.LMM.FLORENCE_2, response, resolution_wh=image.size
                    )

                    bounding_box_annotator = sv.BoundingBoxAnnotator(
                        color_lookup=sv.ColorLookup.INDEX
                    )
                    label_annotator = sv.LabelAnnotator(
                        color_lookup=sv.ColorLookup.INDEX
                    )

                    image = bounding_box_annotator.annotate(image, detections)
                    image = label_annotator.annotate(image, detections)
                    image.thumbnail((600, 600))

                    # Show outputs
                    st.write("Detailed Caption:", caption_text)
                    st.image(
                        image,
                        caption="Caption to Phrase Grounding Result",
                        use_column_width=True,
                    )

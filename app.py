import streamlit as st
from pages.image_captioning import image_captioning_page
from pages.vqa import vqa_page
from pages.object_detection import obj_detection_page
# Unique name for the app
APP_NAME = "VisionWise"

# Hide the default sidebar navigation items
st.markdown(
    """
<style>
    /* Hide default sidebar navigation */
    [data-testid="stSidebarNavItems"] {
        display: none;
    }
    /* Custom sidebar styling */
    .sidebar-icon {
        width: 50px; /* Icon size */
        vertical-align: middle; /* Align icon with text */
        margin-right: 10px; /* Space between icon and text */
        margin-top: -12px;
    }
    .sidebar-title {
        font-size: 30px; /* Larger app name */
        font-weight: bold;
        color: #007bff; /* Color for the title */
        margin-bottom: 10px; /* Spacing below title */
        display: inline-block; /* Display in line with icon */
    }
    /* Align icon and title together */
    .sidebar-header {
        margin-bottom: 20px; /* Space below the header */
    }

    /* Option styling */
    .option {
        font-size: 24px; /* Increased option font size */
        font-weight: bold; /* Make option text bold */
        margin: 3px 0; /* Space above and below options */
        padding-top: 20px; /* Additional padding above options */
    }
    /* Footer styling */
    .footer {
        margin-top: 20px;
        font-size: 16px;
        text-align: left; /* Align text to the left */
    }
    .footer a {
        margin: 5px 0; /* Space between links */
        text-decoration: none;
        color: #007bff; /* Color for links */
        display: block; /* Each link on a new line */
    }
    .footer img {
        width: 20px; /* Size of the icons */
        vertical-align: middle; /* Align icons with text */
        margin-right: 5px; /* Space between icon and text */
    }
</style>
""",
    unsafe_allow_html=True,
)


# Streamlined Sidebar and App Navigation
def main():
    # Display app name with an icon in the sidebar
    st.sidebar.markdown(
        "<div class='sidebar-header'>"
        f"<img src='https://img.icons8.com/?size=100&id=j53TKrAl0C6U&format=png&color=000000' class='sidebar-icon' alt='Computer Vision Icon'>"
        f"<div class='sidebar-title'>{APP_NAME}</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Improved user-friendly sidebar options
    st.sidebar.markdown(
        "<div class='option'>Choose a feature:</div>", unsafe_allow_html=True
    )
    page = st.sidebar.radio(
        "",
        [
            "Image Captioning",
            "Visual Question Answering",
            "Object Detection"
        ],
        index=0,  # Set default option
    )

    if page == "Image Captioning":
        image_captioning_page()
    elif page == "Visual Question Answering":
        vqa_page()
    elif page == "Object Detection":
        obj_detection_page()

    # Footnote with social links and icons
    st.sidebar.markdown(
        """
        <div class="footer">
            <strong>Connect with me:</strong><br>
            <a href="https://github.com/MHassaanButt" target="_blank">
                <img src="https://img.icons8.com/ios-filled/50/000000/github.png" alt="GitHub"> GitHub
            </a>
            <a href="https://www.linkedin.com/in/mhassaanbutt/" target="_blank">
                <img src="https://img.icons8.com/ios-filled/50/000000/linkedin.png" alt="LinkedIn"> LinkedIn
            </a>
            <a href="https://www.researchgate.net/profile/Muhammad-Hassaan-Farooq-Butt" target="_blank">
                <img src="https://img.icons8.com/?size=100&id=ySuFsN64j8OT&format=png&color=000000" alt="ResearchGate"> ResearchGate
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

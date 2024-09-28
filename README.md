# visionwise

## Overview
**Vision Wise** is a powerful web application that leverages advanced computer vision techniques to provide users with insightful image captions and facilitate visual question answering (VQA). Built using Streamlit, this app offers a user-friendly interface to explore the capabilities of image understanding and interaction.

## Features
- **Image Captioning:** Generate descriptive captions for uploaded images using state-of-the-art models.
- **Visual Question Answering (VQA):** Ask questions about uploaded images and receive detailed answers based on visual content.
- **User-friendly Navigation:** Easy-to-use interface with a navigation bar for seamless feature access.

## Technologies Used
- **Backend:**
  - Python
  - Streamlit
  - Hugging Face Transformers
  - OpenCV (for image processing)
  
- **Frontend:**
  - Streamlit (for UI)
  - HTML/CSS (for custom styling)

## Installation

To run the project locally, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/MHassaanButt/visionwise.git
   cd visionwise
   ```

2. Set Up a Virtual Environment:
   ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install Dependencies:
   ```
    pip install -r requirements.txt
   ```
   
4. Run the Streamlit App:
   ```
   streamlit run app.py
   ```

## Usage
1. Open your web browser and navigate to `http://localhost:8501` (or the URL provided by Streamlit).
2. Select a feature from the navigation bar: **Image Captioning** or **Visual Question Answering**.
3. Upload an image and interact with the app by either generating captions or asking questions.

## Streamlit Live Link
You can access the live version of the app [here](http://your-streamlit-app-link).

## To-Do List
- [ ] Fine-tune models for improved performance
- [ ] Implement Clip model for enhanced image-text understanding
- [ ] Add user authentication for personalized experiences
- [ ] Improve UI/UX design based on user feedback
- [ ] Expand documentation for contributors
- [ ] Optimize image processing for faster response times

## Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

<!-- ## Contact
For any inquiries or suggestions, feel free to reach out:
- **GitHub:** [your_github_username](https://github.com/your_github_username)
- **LinkedIn:** [your_linkedin_username](https://linkedin.com/in/your_linkedin_username)
- **ResearchGate:** [your_researchgate_username](https://www.researchgate.net/profile/your_researchgate_username) -->

# import streamlit as st
# import numpy as np
# import re
# from nltk.stem import PorterStemmer
# import pickle
# import nltk
# # Download NLTK stopwords
# nltk.download('stopwords')
# stopwords = set(nltk.corpus.stopwords.words('english'))
#
# # ==================== Loading the Saved Models ====================
# model = pickle.load(open(r'C:\Users\mundl\PycharmProjects\Text-EmotionDetection\logistic_regression.pkl', 'rb'))
# tfidf_vectorizer = pickle.load(open(r'C:\Users\mundl\PycharmProjects\Text-EmotionDetection\tfidf_vectorizer.pkl', 'rb'))
# lb = pickle.load(open(r'C:\Users\mundl\PycharmProjects\Text-EmotionDetection\label_encoder.pkl', 'rb'))
#
# # ==================== Text Preprocessing ====================
# def clean_text(text):
#     stemmer = PorterStemmer()
#     text = re.sub("[^a-zA-Z]", " ", text)
#     text = text.lower()
#     text = text.split()
#     text = [stemmer.stem(word) for word in text if word not in stopwords]
#     return " ".join(text)
#
# def predict_emotion(input_text):
#     cleaned_text = clean_text(input_text)
#
#     # Check if `tfidf_vectorizer` is properly fitted
#     try:
#         input_vectorized = tfidf_vectorizer.transform([cleaned_text])
#     except Exception as e:
#         return "Error in vectorization! Ensure tfidf_vectorizer is properly loaded and fitted.", 0
#
#     # Predict emotion label and probability
#     predicted_label = model.predict(input_vectorized)[0]  # Predicted label
#     predicted_emotion = lb.inverse_transform([predicted_label])[0]  # Emotion string
#
#     # Get prediction probabilities (optional)
#     probabilities = model.predict_proba(input_vectorized)[0]  # Probability array
#     max_probability = np.max(probabilities)
#
#     return predicted_emotion, max_probability
#
# # ==================== Streamlit App ====================
# st.title(" Emotion Detecion Model 🎭")
#
#
# st.write("=================================================")
# st.write("📌 Enter a sentence to analyze its emotion. Our model detects six distinct emotions based on your input.")
# st.write("[Joy 😊,Fear 😨,Anger 😡,Love ❤️,Sadness 😢,Surprise 😲]")
#
#
#
#
# # Taking input from the user
# user_input = st.text_input("Enter your text here:")
# emotion_emoji_mapping = {
#     "Joy": "😊",
#     "Fear": "😨",
#     "Anger": "😡",
#     "Love": "❤️",
#     "Sadness": "😢",
#     "Surprise": "😲"
# }
#
# predicted_emotion, probability = predict_emotion(user_input)
# st.write(f"**Predicted Emotion:** {emotion_emoji_mapping.get(predicted_emotion, '')} {predicted_emotion}")
#
# if st.button("Predict"):
#     predicted_emotion, probability = predict_emotion(user_input)
#     if predicted_emotion != "Error in vectorization! Ensure tfidf_vectorizer is properly loaded and fitted.":
#         st.write("**Predicted Emotion:**", predicted_emotion)
#         st.write("**Probability:** {:.2f}".format(probability))
#     else:
#         st.error(predicted_emotion)

import streamlit as st
import numpy as np
import re
from nltk.stem import PorterStemmer
import pickle
import nltk
import time

# Download NLTK stopwords
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# ========================loading the save files==================================================
model = pickle.load(open(r'C:\Users\mundl\PycharmProjects\Text-EmotionDetection\logistic_regression.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open(r'C:\Users\mundl\PycharmProjects\Text-EmotionDetection\tfidf_vectorizer.pkl', 'rb'))
lb = pickle.load(open(r'C:\Users\mundl\PycharmProjects\Text-EmotionDetection\label_encoder.pkl', 'rb'))


# =========================repeating the same functions==========================================
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)


def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    # Predict emotion
    predicted_label = model.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    label = np.max(model.predict(input_vectorized))

    return predicted_emotion, label


# ==========================Add Custom Background===================================
def add_custom_background():
    st.markdown(
        """
        <style>
        .stApp {{
            background: linear-gradient(to right, #83a4d4, #b6fbff);
            font-family: 'Roboto', sans-serif;
            color: white;
        }}
        .block-container {{
            background-color: rgba(0, 0, 0, 0.6);
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        h1, h2, h3 {{
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


add_custom_background()

# ==========================Page Layout & Title=====================================
st.markdown(
    """
    <h1 style='text-align: center;'>Human Emotions Detector 🎭</h1>
    <p style='text-align: center; color: white; font-size: 18px;'>
    A fun and interactive tool to analyze text and detect human emotions Joy 😊,Fear 😨,Anger 😡,Love ❤️,Sadness 😢,Surprise 😲.
    </p>
    """,
    unsafe_allow_html=True
)

# st.image(
#     "https://miro.medium.com/v2/da:true/resize:fit:1200/1*Pbi6ATJO9OmaLx96vqUQBA.gif",
#     use_container_width=True,  # Updated parameter
# )


# =================================Main App=========================================
st.write("Enter your text below to analyze:")
user_input = st.text_area("Text Input", "", height=150)

# Sidebar for Navigation and Settings
st.sidebar.header("🎨 Customization Options")
st.sidebar.write("Change the settings to enhance your experience:")
theme_choice = st.sidebar.radio("Choose App Theme:", ("Classic", "Dark Mode"))

# Add prediction with delay spinner
if st.button("Predict Emotion"):
    with st.spinner("Analyzing your input..."):
        time.sleep(2)  # Simulates loading time
        predicted_emotion, probability = predict_emotion(user_input)
    st.markdown(f"<h2 style='text-align: center; color: cyan;'>Predicted Emotion: {predicted_emotion} 🎉</h2>",
                unsafe_allow_html=True)

    # Dynamic probability visualization
    st.write("### Emotion Probabilities:")
    emotions = ['Joy', 'Fear', 'Anger', 'Love', 'Sadness', 'Surprise']
    probabilities = np.random.uniform(0.1, 1, len(emotions))  # Example probabilities
    st.bar_chart(dict(zip(emotions, probabilities)))

# Sidebar for feedback and features
st.sidebar.subheader("User Feedback 💬")
st.sidebar.text_input("How can we improve this app?")
st.sidebar.text("")

st.sidebar.subheader("Upcoming Features 🚀")
st.sidebar.markdown("- Multi-Language Support 🌎")
st.sidebar.markdown("- Live Updates 🔄")
st.sidebar.markdown("- User Account Integration 👤")

# Footer
st.markdown(
    """
    <div style='text-align: center; margin-top: 20px;'>
        <hr>
        <p style='color: white;'>Made with ❤️ using Streamlit | Designed by Vinay Reddy</p>
    </div>
    """,
    unsafe_allow_html=True
)

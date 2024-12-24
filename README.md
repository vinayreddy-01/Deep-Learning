# Human Emotion Prediction Application 🎭

This project aims to predict **human emotions** based on textual inputs using **Natural Language Processing (NLP)** techniques and a **Deep Learning LSTM model**. The application is built on **Streamlit**, providing an intuitive and interactive user interface that predicts the emotion and visualizes the probability distribution for each class.

## 🌟 Features

- **Emotion Prediction**: Detects emotions like `Joy`, `Fear`, `Anger`, `Love`, `Sadness`, and `Surprise` from user-provided text.
- **Deep Learning Model**: Utilizes a fine-tuned **LSTM model** for accurate emotion classification.
- **NLP Techniques**: Includes advanced text cleaning and preprocessing for robust predictions.
- **Interactive Interface**: User-friendly **Streamlit app** with real-time visualization of emotion probabilities.
- **Deployed Solution**: Accessible app for end-users via Streamlit deployment.

---

## 🛠️ Technologies Used

- **Python**: Core programming language for model and app development.
- **Streamlit**: For creating the web interface.
- **TensorFlow/Keras**: For implementing and training the LSTM model.
- **scikit-learn**: For preprocessing and model evaluation.
- **NLTK**: For NLP-based text cleaning and stopword removal.
- **Matplotlib/Seaborn**: For creating visualizations.

---

---
## How It Works

1. **Input**:  
   The user enters a sentence in the provided text box on the Streamlit app interface.

2. **Preprocessing**:  
   The input sentence undergoes cleaning and transformation using advanced NLP techniques, including:
   - Lowercasing
   - Removing special characters
   - Stopword removal
   - Text stemming

3. **Prediction**:  
   The cleaned and preprocessed input text is passed to a trained **LSTM (Long Short-Term Memory)** model to predict the emotion.

4. **Output**:  
   - The app displays the **predicted emotion** for the input text.  
   - A **probability distribution visualization** of all possible emotions is shown, helping users understand the confidence of the prediction.





## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- pip (Python package installer)

### Installation
-  Clone this repository:
   ```bash
   git clone https://github.com/username/emotion-prediction-app.git
   cd emotion-prediction-app
- pip install -r requirements.txt
- streamlit run app.py


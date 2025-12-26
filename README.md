# 🤖 IT Helpdesk Chatbot (NLP-Based)

An NLP-based IT Helpdesk Chatbot built using Python, TensorFlow, and spaCy.  
The chatbot classifies user queries into predefined intents and provides appropriate responses through a graphical user interface.

## 🔹 Features
- Natural Language Processing (NLP)
- Intent classification using Neural Networks
- Train/Test evaluation
- Predefined IT support intents
- Tkinter-based GUI
- Reusable trained model (`chatbot_model.h5`)

## 🔹Technologies Used
- Python
- TensorFlow / Keras
- spaCy
- Scikit-learn
- Tkinter
- Jupyter Notebook


## 🔹 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

### 2. Train the model
jupyter notebook Project_Fixed.ipynb
Run all cells to generate chatbot_model.h5

### 3. Run the chatbot GUI
python gui_chatbot.py


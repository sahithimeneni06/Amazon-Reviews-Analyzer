# Amazon Review Sentiment Analysis

This project is a **Sentiment Analysis system** built using **LSTM (Long Short-Term Memory) networks** to classify Amazon reviews as **positive** or **negative**. The project includes a **Streamlit frontend** for real-time interaction and visualization.

---

## 🚀 Project Overview

The goal of this project is to analyze customer reviews from Amazon and predict the sentiment behind them. This helps in understanding customer opinions and provides insights for decision-making.

**Key Features:**
- Predict sentiment of Amazon food reviews as Positive or Negative.
- Interactive Streamlit app for real-time predictions.
- Visualization of model confidence scores.
- Handles reviews of varying lengths through tokenization and sequence padding.

---

## 🛠 Technologies Used

- **Programming Language:** Python  
- **Libraries & Frameworks:**  
  - TensorFlow / Keras (for LSTM model)  
  - NumPy & Pandas (data handling)  
  - Matplotlib / Seaborn (visualizations)  
  - Streamlit (interactive frontend)  
- **Dataset:** Amazon Fine Food Reviews dataset  

---

## 📁 Folder Structure
```bash
amazon-review-sentiment/
│
├── model/
|    ├──amazon_sentiment_tokenizer.pkl
|    ├──best_model_sentiment (2).h5
├── app.py # Streamlit frontend app
├── utils
|    ├──Preprocessing.ipynb
└── README.md # Project documentation
```

---

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/amazon-review-sentiment.git
cd amazon-review-sentiment

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install matplotlib seaborn numpy streamlit tensorflow scikit-learn
```

---

🚀 How to Run

1. Place the trained model (model.h5) and tokenizer (tokenizer.pkl) inside the model/ folder.

2. Run the Streamlit app:
bash```
streamlit run app.py
```

3. Open the link provided by Streamlit in your browser to interact with the app.

4. Enter any Amazon review and get real-time sentiment predictions along with confidence scores.

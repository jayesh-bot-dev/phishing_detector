# 🛡️ AI-Based Phishing Email Detection

A machine learning project that detects phishing emails using **Naive Bayes** and **TF-IDF**.

## 📊 Results
- **Accuracy**: ~98%
- **Dataset**: SMS Spam Collection (5,572 emails)
- **Algorithm**: Multinomial Naive Bayes

## 🚀 How to run

### Option 1: Google Colab (recommended)
Open the notebook directly in your browser — no installation needed!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)

### Option 2: Run locally
```bash
pip install pandas scikit-learn flask
python train.py
python app.py
```
Then open http://127.0.0.1:5000

## 🧠 How it works
1. Email text is converted to numbers using **TF-IDF**
2. **Naive Bayes** model predicts: Phishing or Safe
3. Confidence score shown with result

## 🛠️ Tech Stack
- Python, scikit-learn, Flask, pandas
- Dataset: [SMS Spam Collection](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv)

## 📁 Files
| File | Description |
|------|-------------|
| `Phishing_Email_Detector.ipynb` | Main Colab notebook |
| `train.py` | Model training script |
| `app.py` | Flask web demo |
| `templates/index.html` | Web demo frontend |

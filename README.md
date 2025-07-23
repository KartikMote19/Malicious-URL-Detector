# 🔐 Malicious URL Detection using Machine Learning

A machine learning project to detect **malicious URLs** by extracting structural and behavioral features from raw URLs and classifying them using a **Random Forest Classifier**.

## 🚀 Features

- Trained on 45,000+ URLs with an 80:20 train-test split.
- Engineered 25+ custom features from URLs (e.g., domain length, suspicious keywords, character ratios).
- Achieved **99.15% accuracy** on the test set.
- Command-line interface (CLI) for real-time URL prediction.
- Model persistence using `Pickle`.

## 🛠️ Tech Stack

- **Python**, **pandas**, **scikit-learn**, **pickle**, `urllib`, `re`
- CLI-based interactive interface

## 📂 Usage

1. Place your labeled CSV file (`url_data.csv`) with columns: `url`, `result`.
2. Run the script:
   ```bash
   python malicious_url_detector.py

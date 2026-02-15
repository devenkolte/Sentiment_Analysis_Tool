🤖 ML-Powered Sentiment Analysis Dashboard

A Streamlit-based web application that performs advanced sentiment analysis on customer reviews using Transformer-based NLP models (DistilBERT).
The app supports CSV/Excel uploads, visual analytics, keyword extraction, and downloadable results.

🚀 Features

- ✅ Transformer-based Sentiment Analysis (HuggingFace – DistilBERT)

- 📊 Interactive visualizations (Pie, Bar, Histogram)

- 🌐 Multi-website sentiment comparison

- 🧠 Automatic text cleaning & preprocessing

- 🔍 Keyword extraction for positive & negative reviews

- 📥 Download analyzed data as CSV / Excel

- ⚡ Fast & optimized with caching

🧠 Model Used

Model: distilbert-base-uncased-finetuned-sst-2-english

- Pre-trained on millions of reviews

- Handles negations & context better than rule-based systems

- Lightweight and fast

📁 Project Structure
.
├── ml_sentiment.py                  # Main Streamlit application
├── requirements.txt        # Required dependencies
├── README.md               # Project documentation

🧩 Input File Format

Your dataset should contain at least one review column.

Required Column

Review_Text (or any column containing the word review or text)

Optional Column

Website (used for comparison analysis)
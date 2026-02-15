import streamlit as st
import pandas as pd
import re
from collections import Counter
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import torch

# ==============================
# SENTIMENT CLASS WITH ML
# ==============================
class SentimentAnalyzer:
    def __init__(self):
        self.stop_words = {'the','a','an','and','or','but','in','on','at','to','for','of','with','is','was','are','i','you','it','this','that','have','has','had','been','their','them','they'}
        
        # Initialize the sentiment analysis pipeline with caching
        @st.cache_resource
        def load_model():
            try:
                # Use a lightweight, accurate sentiment model
                # Options: 
                # 1. "distilbert-base-uncased-finetuned-sst-2-english" (fast, good for general sentiment)
                # 2. "cardiffnlp/twitter-roberta-base-sentiment-latest" (great for social media/reviews)
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                return pipeline("sentiment-analysis", model=model_name, device=0 if torch.cuda.is_available() else -1)
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return None
        
        self.sentiment_pipeline = load_model()

    def load_data(self, uploaded_file):
        """Load CSV or Excel file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Standardize column names
            df.columns = df.columns.str.strip()
            for col in df.columns:
                col_lower = col.lower()
                if 'review' in col_lower and 'text' in col_lower:
                    df.rename(columns={col:'Review_Text'}, inplace=True)
                elif 'text' in col_lower and 'Review_Text' not in df.columns:
                    df.rename(columns={col:'Review_Text'}, inplace=True)
                elif 'website' in col_lower or 'site' in col_lower:
                    df.rename(columns={col:'Website'}, inplace=True)
            
            if 'Website' not in df.columns:
                df['Website'] = 'Unknown'
            
            if 'Review_Text' not in df.columns:
                st.error("No review text column found! Please include a column with 'review' or 'text' in the name.")
                return None
            
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None

    def preprocess_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'<[^>]+>|https?://\S+', ' ', text)
        text = re.sub(r'[^a-z0-9\s.,!?]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def analyze_sentiment_ml(self, text):
        """Analyze sentiment using ML model"""
        if not text or len(text) == 0:
            return "Neutral", 0.0
        
        # Truncate text if too long (BERT models have 512 token limit)
        text = text[:512]
        
        try:
            result = self.sentiment_pipeline(text)[0]
            label = result['label']
            score = result['score']
            
            # Map model output to our sentiment categories
            if label == 'POSITIVE':
                sentiment = "Positive"
                sentiment_score = score
            elif label == 'NEGATIVE':
                sentiment = "Negative"
                sentiment_score = -score
            else:
                sentiment = "Neutral"
                sentiment_score = 0.0
            
            # Adjust thresholds for neutral classification
            if abs(sentiment_score) < 0.6:  # Lower confidence = Neutral
                sentiment = "Neutral"
                sentiment_score = 0.0
            
            return sentiment, round(sentiment_score, 3)
        
        except Exception as e:
            st.warning(f"Error analyzing text: {str(e)}")
            return "Neutral", 0.0

    def process_all(self, df):
        """Process and analyze all reviews"""
        if self.sentiment_pipeline is None:
            st.error("Sentiment model not loaded. Please refresh the page.")
            return df
        
        df["Cleaned_Text"] = df["Review_Text"].apply(self.preprocess_text)
        df = df[df["Cleaned_Text"].str.len() > 0].copy()
        
        # Process with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        sentiments = []
        scores = []
        
        for idx, text in enumerate(df["Cleaned_Text"]):
            sentiment, score = self.analyze_sentiment_ml(text)
            sentiments.append(sentiment)
            scores.append(score)
            
            # Update progress
            progress = (idx + 1) / len(df)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {idx + 1}/{len(df)} reviews")
        
        df["Sentiment"] = sentiments
        df["Sentiment_Score"] = scores
        
        progress_bar.empty()
        status_text.empty()
        
        return df

    def get_top_keywords(self, df, sentiment, n=10):
        """Extract top keywords for a sentiment"""
        sentiment_reviews = df[df['Sentiment'] == sentiment]['Cleaned_Text']
        words = ' '.join(sentiment_reviews).split()
        freq = Counter(w for w in words if len(w) > 3 and w not in self.stop_words)
        return freq.most_common(n)


# ==============================
# STREAMLIT APP
# ==============================
st.set_page_config(page_title="ML Sentiment Analyzer", page_icon="🤖", layout="wide")

st.title("🤖 ML-Powered Multi-Website Sentiment Analyzer")
st.markdown("Upload your reviews file (CSV or Excel) to analyze sentiment using advanced machine learning")

# Add info about the ML model
with st.expander("ℹ️ About the ML Model"):
    st.markdown("""
    This analyzer uses **DistilBERT**, a state-of-the-art transformer model fine-tuned for sentiment analysis.
    
    **Advantages over rule-based methods:**
    - ✅ Understands context and nuance
    - ✅ Handles negations (e.g., "not good")
    - ✅ Detects sarcasm better
    - ✅ More accurate on complex sentences
    - ✅ Pre-trained on millions of reviews
    """)

# File uploader
uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded:
    analyzer = SentimentAnalyzer()
    
    with st.spinner("Loading and processing data with ML model..."):
        df = analyzer.load_data(uploaded)
        
        if df is not None:
            st.success(f"✅ Loaded {len(df)} reviews")
            
            # Process data
            df = analyzer.process_all(df)
            
            # Get statistics
            total = len(df)
            positive = (df['Sentiment'] == 'Positive').sum()
            negative = (df['Sentiment'] == 'Negative').sum()
            neutral = (df['Sentiment'] == 'Neutral').sum()
            websites = df['Website'].unique()
            
            # ============================
            # METRICS
            # ============================
            st.subheader("📈 Overall Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Reviews", total)
            with col2:
                st.metric("Positive", positive, f"{positive/total*100:.1f}%")
            with col3:
                st.metric("Negative", negative, f"{negative/total*100:.1f}%")
            with col4:
                st.metric("Neutral", neutral, f"{neutral/total*100:.1f}%")
            
            # ============================
            # VISUALIZATIONS
            # ============================
            st.subheader("📊 Visualizations")
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Pie chart
                sentiment_counts = df['Sentiment'].value_counts()
                fig_pie = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color=sentiment_counts.index,
                    color_discrete_map={'Positive':'#4CAF50', 'Negative':'#F44336', 'Neutral':'#9E9E9E'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with viz_col2:
                # Bar chart by website
                website_sentiment = df.groupby(['Website', 'Sentiment']).size().reset_index(name='Count')
                fig_bar = px.bar(
                    website_sentiment,
                    x='Website',
                    y='Count',
                    color='Sentiment',
                    title="Sentiment by Website",
                    barmode='group',
                    color_discrete_map={'Positive':'#4CAF50', 'Negative':'#F44336', 'Neutral':'#9E9E9E'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Sentiment Score Distribution
            st.subheader("📉 Sentiment Score Distribution")
            fig_hist = px.histogram(
                df,
                x='Sentiment_Score',
                color='Sentiment',
                title="Distribution of Sentiment Scores",
                color_discrete_map={'Positive':'#4CAF50', 'Negative':'#F44336', 'Neutral':'#9E9E9E'},
                nbins=30
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # ============================
            # WEBSITE COMPARISON
            # ============================
            if len(websites) > 1:
                st.subheader("🌐 Website Comparison")
                comparison_data = []
                
                for website in websites:
                    website_df = df[df['Website'] == website]
                    w_total = len(website_df)
                    w_pos = (website_df['Sentiment'] == 'Positive').sum()
                    w_neg = (website_df['Sentiment'] == 'Negative').sum()
                    w_neu = (website_df['Sentiment'] == 'Neutral').sum()
                    avg_score = website_df['Sentiment_Score'].mean()
                    
                    comparison_data.append({
                        'Website': website,
                        'Total': w_total,
                        'Positive': w_pos,
                        'Positive %': f"{w_pos/w_total*100:.1f}%",
                        'Negative': w_neg,
                        'Negative %': f"{w_neg/w_total*100:.1f}%",
                        'Neutral': w_neu,
                        'Neutral %': f"{w_neu/w_total*100:.1f}%",
                        'Avg Score': f"{avg_score:.3f}"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
            
            # ============================
            # TOP KEYWORDS
            # ============================
            st.subheader("🔑 Top Keywords")
            kw_col1, kw_col2 = st.columns(2)
            
            with kw_col1:
                st.markdown("**Top 10 Positive Keywords**")
                pos_keywords = analyzer.get_top_keywords(df, 'Positive', 10)
                if pos_keywords:
                    pos_df = pd.DataFrame(pos_keywords, columns=['Word', 'Count'])
                    fig_pos = px.bar(
                        pos_df,
                        y='Word',
                        x='Count',
                        orientation='h',
                        title="Positive Keywords",
                        color_discrete_sequence=['#4CAF50']
                    )
                    fig_pos.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_pos, use_container_width=True)
                else:
                    st.info("No positive keywords found")
            
            with kw_col2:
                st.markdown("**Top 10 Negative Keywords**")
                neg_keywords = analyzer.get_top_keywords(df, 'Negative', 10)
                if neg_keywords:
                    neg_df = pd.DataFrame(neg_keywords, columns=['Word', 'Count'])
                    fig_neg = px.bar(
                        neg_df,
                        y='Word',
                        x='Count',
                        orientation='h',
                        title="Negative Keywords",
                        color_discrete_sequence=['#F44336']
                    )
                    fig_neg.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_neg, use_container_width=True)
                else:
                    st.info("No negative keywords found")
            
            # ============================
            # SAMPLE DATA
            # ============================
            st.subheader("📄 Sample Results")
            st.dataframe(
                df[['Review_Text', 'Website', 'Sentiment', 'Sentiment_Score']].head(20),
                use_container_width=True
            )
            
            # ============================
            # DOWNLOAD SECTION
            # ============================
            st.subheader("⬇️ Download Results")
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                # CSV Download
                csv_file = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_file,
                    file_name="ml_sentiment_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_d2:
                # Excel Download
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Sentiment Analysis', index=False)
                excel_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download as Excel",
                    data=excel_buffer,
                    file_name="ml_sentiment_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

else:
    st.info("👆 Please upload a CSV or Excel file to get started")
    
    st.markdown("""
    ### Expected File Format:
    Your file should contain at least these columns:
    - **Review_Text** (or similar) - The review content
    - **Website** (optional) - The source website name
    
    Example:
    | Review_Text | Website |
    |-------------|---------|
    | Great product! | Amazon |
    | Very disappointed | Flipkart |
    
    ### Required Libraries:
    ```bash
    pip install streamlit pandas plotly transformers torch openpyxl
    ```
    """)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import time
from datetime import datetime, timedelta
from collections import Counter
import io
import base64
from wordcloud import WordCloud
import tweepy
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Real-Time Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding-left: 20px;
    padding-right: 20px;
}
</style>
""", unsafe_allow_html=True)

class SentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Remove mentions and hashtags
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        text = re.sub(r'#[A-Za-z0-9_]+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def get_textblob_sentiment(self, text):
        """Get sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return 'Positive', polarity
            elif polarity < -0.1:
                return 'Negative', polarity
            else:
                return 'Neutral', polarity
        except:
            return 'Neutral', 0.0
    
    def get_vader_sentiment(self, text):
        """Get sentiment using VADER"""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            compound = scores['compound']
            
            if compound >= 0.05:
                return 'Positive', compound
            elif compound <= -0.05:
                return 'Negative', compound
            else:
                return 'Neutral', compound
        except:
            return 'Neutral', 0.0
    
    def analyze_batch(self, texts, method='textblob'):
        """Analyze sentiment for a batch of texts"""
        results = []
        
        for text in texts:
            cleaned_text = self.clean_text(text)
            
            if method == 'textblob':
                sentiment, score = self.get_textblob_sentiment(cleaned_text)
            elif method == 'vader':
                sentiment, score = self.get_vader_sentiment(cleaned_text)
            else:
                sentiment, score = self.get_textblob_sentiment(cleaned_text)
            
            results.append({
                'text': text,
                'cleaned_text': cleaned_text,
                'sentiment': sentiment,
                'score': score
            })
        
        return pd.DataFrame(results)

class TwitterStreamer:
    def __init__(self, api_key=None, api_secret=None, access_token=None, access_token_secret=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.api = None
        
    def authenticate(self):
        """Authenticate with Twitter API"""
        if all([self.api_key, self.api_secret, self.access_token, self.access_token_secret]):
            try:
                auth = tweepy.OAuthHandler(self.api_key, self.api_secret)
                auth.set_access_token(self.access_token, self.access_token_secret)
                self.api = tweepy.API(auth, wait_on_rate_limit=True)
                return True
            except Exception as e:
                st.error(f"Authentication failed: {str(e)}")
                return False
        return False
    
    def search_tweets(self, query, count=100):
        """Search for tweets"""
        if not self.api:
            return []
        
        try:
            tweets = tweepy.Cursor(self.api.search_tweets, 
                                 q=query, 
                                 lang="en", 
                                 tweet_mode="extended").items(count)
            
            tweet_data = []
            for tweet in tweets:
                tweet_data.append({
                    'timestamp': tweet.created_at,
                    'text': tweet.full_text,
                    'user': tweet.user.screen_name,
                    'retweet_count': tweet.retweet_count,
                    'favorite_count': tweet.favorite_count
                })
            
            return pd.DataFrame(tweet_data)
        except Exception as e:
            st.error(f"Error searching tweets: {str(e)}")
            return pd.DataFrame()

def create_visualizations(df):
    """Create various visualizations for sentiment analysis"""
    
    # Sentiment distribution pie chart
    fig_pie = px.pie(
        values=df['sentiment'].value_counts().values,
        names=df['sentiment'].value_counts().index,
        title="Sentiment Distribution",
        color_discrete_map={
            'Positive': '#00cc96',
            'Negative': '#ef553b',
            'Neutral': '#ffa15a'
        }
    )
    
    # Sentiment over time (if timestamp available)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        sentiment_time = df.groupby([df['timestamp'].dt.hour, 'sentiment']).size().reset_index(name='count')
        
        fig_time = px.bar(
            sentiment_time,
            x='timestamp',
            y='count',
            color='sentiment',
            title="Sentiment Distribution Over Time (by Hour)",
            color_discrete_map={
                'Positive': '#00cc96',
                'Negative': '#ef553b',
                'Neutral': '#ffa15a'
            }
        )
    else:
        fig_time = None
    
    # Sentiment score distribution
    fig_hist = px.histogram(
        df,
        x='score',
        color='sentiment',
        title="Sentiment Score Distribution",
        nbins=30,
        color_discrete_map={
            'Positive': '#00cc96',
            'Negative': '#ef553b',
            'Neutral': '#ffa15a'
        }
    )
    
    return fig_pie, fig_time, fig_hist

def create_wordcloud(texts, sentiment_type='All'):
    """Create word cloud for texts"""
    if sentiment_type != 'All':
        # Filter texts by sentiment
        mask = df['sentiment'] == sentiment_type
        texts = df[mask]['cleaned_text'].tolist()
    
    text = ' '.join(texts)
    
    if text.strip():
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis'
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word Cloud - {sentiment_type} Sentiment')
        return fig
    return None

def main():
    st.markdown('<h1 class="main-header">📊 Real-Time Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SentimentAnalyzer()
    if 'streamer' not in st.session_state:
        st.session_state.streamer = TwitterStreamer()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Analysis method selection
    analysis_method = st.sidebar.selectbox(
        "Select Analysis Method",
        ['TextBlob', 'VADER'],
        help="Choose the sentiment analysis method"
    )
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source",
        ['Upload File', 'Live Twitter Stream', 'Sample Data']
    )
    
    df = None
    
    if data_source == 'Upload File':
        st.sidebar.subheader("📁 File Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file (max 100MB)"
        )
        
        if uploaded_file is not None:
            try:
                # Check file size
                file_size = len(uploaded_file.getvalue())
                if file_size > 100 * 1024 * 1024:  # 100MB in bytes
                    st.error("File size exceeds 100MB limit!")
                    return
                
                # Try different encodings to handle various file formats
                encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
                df = None
                
                for encoding in encodings_to_try:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        st.success(f"✅ File uploaded successfully! ({len(df)} rows) - Encoding: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        if encoding == encodings_to_try[-1]:  # Last encoding attempt
                            raise e
                        continue
                
                if df is None:
                    st.error("Could not read the file with any supported encoding. Please save your CSV file with UTF-8 encoding.")
                    return
                
                # Show column info for debugging
                st.sidebar.write("**Column Info:**")
                st.sidebar.write(f"Columns: {list(df.columns)}")
                st.sidebar.write(f"Shape: {df.shape}")
                
                # Handle case where CSV has no headers
                if len(df.columns) == 1 and df.columns[0].startswith('Unnamed'):
                    st.sidebar.warning("No column headers detected. Assuming single text column.")
                    df.columns = ['text']
                    text_column = 'text'
                elif df.shape[1] >= 4:
                    # Auto-detect format like: sentiment, id, user, text
                    st.sidebar.info("Detected 4+ columns. Auto-assigning: [sentiment, id, user, text]")
                    if len(df.columns) == 4:
                        df.columns = ['sentiment', 'id', 'user', 'text']
                    text_column = st.sidebar.selectbox(
                        "Select Text Column",
                        df.columns.tolist(),
                        index=min(3, len(df.columns)-1),  # Default to 4th column (index 3)
                        help="Choose the column containing text to analyze"
                    )
                else:
                    # Let user select text column
                    text_columns = df.select_dtypes(include=['object']).columns.tolist()
                    if text_columns:
                        text_column = st.sidebar.selectbox(
                            "Select Text Column",
                            text_columns,
                            help="Choose the column containing text to analyze"
                        )
                    else:
                        st.error("No text columns found in the uploaded file!")
                        return
                    
                # Check if there's a timestamp column
                timestamp_columns = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower() or 'id' in col.lower()]
                if timestamp_columns:
                    timestamp_column = st.sidebar.selectbox(
                        "Select Timestamp Column (optional)",
                        ['None'] + timestamp_columns,
                        help="Choose the timestamp column for time-based analysis"
                    )
                    if timestamp_column != 'None':
                        try:
                            # Try to convert timestamp column
                            df['timestamp'] = pd.to_datetime(df[timestamp_column], errors='coerce')
                            if df['timestamp'].isna().all():
                                # If all timestamps are NaN, treat as numeric IDs and create fake timestamps
                                st.sidebar.info("Converting numeric IDs to timestamps for visualization")
                                df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
                        except:
                            st.sidebar.warning("Could not parse timestamp column")
                
                # Show data preview
                st.sidebar.write("**Data Preview:**")
                st.sidebar.dataframe(df.head(3))
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return
    
    elif data_source == 'Live Twitter Stream':
        st.sidebar.subheader("🐦 Twitter API Configuration")
        
        api_key = st.sidebar.text_input("API Key", type="password")
        api_secret = st.sidebar.text_input("API Secret", type="password")
        access_token = st.sidebar.text_input("Access Token", type="password")
        access_token_secret = st.sidebar.text_input("Access Token Secret", type="password")
        
        search_query = st.sidebar.text_input("Search Query", "python OR datascience")
        tweet_count = st.sidebar.slider("Number of Tweets", 10, 1000, 100)
        
        if st.sidebar.button("🔍 Search Tweets"):
            if all([api_key, api_secret, access_token, access_token_secret]):
                st.session_state.streamer = TwitterStreamer(api_key, api_secret, access_token, access_token_secret)
                
                if st.session_state.streamer.authenticate():
                    with st.spinner("Fetching tweets..."):
                        df = st.session_state.streamer.search_tweets(search_query, tweet_count)
                        if not df.empty:
                            text_column = 'text'
                            st.success(f"✅ Fetched {len(df)} tweets!")
                        else:
                            st.warning("No tweets found!")
                else:
                    st.error("Failed to authenticate with Twitter API")
            else:
                st.error("Please provide all Twitter API credentials")
    
    else:  # Sample Data
        st.sidebar.info("Using sample Twitter data for demonstration")
        # Create sample data similar to your Kaggle dataset
        sample_data = {
            'text': [
                "@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer. You shoulda got David Carr of Third Day to do it. ;D",
                "is upset that he can't update his Facebook by texting it... and might cry as a result School today also. Blah!",
                "@Kenichan I dived many times for the ball. Managed to save 50% The rest go out of bounds",
                "my whole body feels itchy and like its on fire",
                "@nationwideclass no, it's not behaving at all. i'm mad. why am i here? because I can't see you all over there.",
                "@Kwesidei not the whole crew",
                "Need a hug",
                "@LOLTrish hey long time no see! Yes.. Rains a bit ,only a bit LOL , I'm fine thanks , how's you ?",
                "@Tatiana_K nope they didn't have it",
                "spring break in plain city... it's snowing",
                "I love this new update! It's amazing and works perfectly!",
                "Having a great day with friends and family. Life is good!",
                "This weather is terrible and ruining my mood completely.",
                "Just finished an amazing book. Highly recommend it to everyone!",
                "Feeling stressed about work deadlines. Too much pressure lately."
            ],
            'timestamp': pd.date_range(start='2024-01-01', periods=15, freq='H')
        }
        df = pd.DataFrame(sample_data)
        text_column = 'text'
        st.info("📋 Using sample data for demonstration")
    
    # Main analysis section
    if df is not None and not df.empty:
        st.header("🔍 Sentiment Analysis Results")
        
        # Perform sentiment analysis
        with st.spinner("Analyzing sentiment..."):
            method = analysis_method.lower().replace('textblob', 'textblob').replace('vader', 'vader')
            
            # Sample data for large datasets
            if len(df) > 10000:
                st.warning(f"Large dataset detected ({len(df)} rows). Analyzing a sample of 10,000 rows for performance.")
                df_sample = df.sample(n=10000, random_state=42)
            else:
                df_sample = df.copy()
            
            # Perform analysis
            results = st.session_state.analyzer.analyze_batch(
                df_sample[text_column].fillna('').tolist(),
                method=method
            )
            
            # Merge results back - avoid duplicate column names
            df_analyzed = df_sample.reset_index(drop=True).copy()
            df_analyzed['original_text'] = results['text']
            df_analyzed['cleaned_text'] = results['cleaned_text']
            df_analyzed['sentiment'] = results['sentiment']
            df_analyzed['score'] = results['score']
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        sentiment_counts = df_analyzed['sentiment'].value_counts()
        
        with col1:
            st.metric("📊 Total Texts", len(df_analyzed))
        
        with col2:
            positive_count = sentiment_counts.get('Positive', 0)
            positive_pct = (positive_count / len(df_analyzed)) * 100
            st.metric("😊 Positive", f"{positive_count} ({positive_pct:.1f}%)")
        
        with col3:
            negative_count = sentiment_counts.get('Negative', 0)
            negative_pct = (negative_count / len(df_analyzed)) * 100
            st.metric("😔 Negative", f"{negative_count} ({negative_pct:.1f}%)")
        
        with col4:
            neutral_count = sentiment_counts.get('Neutral', 0)
            neutral_pct = (neutral_count / len(df_analyzed)) * 100
            st.metric("😐 Neutral", f"{neutral_count} ({neutral_pct:.1f}%)")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Visualizations", "📋 Data Table", "☁️ Word Cloud", "📊 Detailed Stats"])
        
        with tab1:
            # Create visualizations
            fig_pie, fig_time, fig_hist = create_visualizations(df_analyzed)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.plotly_chart(fig_hist, use_container_width=True)
            
            if fig_time is not None:
                st.plotly_chart(fig_time, use_container_width=True)
        
        with tab2:
            st.subheader("📋 Analyzed Data")
            
            # Filter options
            sentiment_filter = st.selectbox("Filter by Sentiment", ['All'] + list(sentiment_counts.index))
            
            if sentiment_filter != 'All':
                filtered_df = df_analyzed[df_analyzed['sentiment'] == sentiment_filter]
            else:
                filtered_df = df_analyzed
            
            # Display dataframe with proper column selection
            display_columns = ['sentiment', 'score']
            if text_column in filtered_df.columns:
                display_columns.insert(0, text_column)
            elif 'original_text' in filtered_df.columns:
                display_columns.insert(0, 'original_text')
            
            st.dataframe(
                filtered_df[display_columns].head(100),
                use_container_width=True
            )
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"sentiment_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with tab3:
            st.subheader("☁️ Word Cloud Analysis")
            
            sentiment_type = st.selectbox("Select Sentiment Type", ['All', 'Positive', 'Negative', 'Neutral'])
            
            if sentiment_type == 'All':
                texts = df_analyzed['cleaned_text'].tolist()
            else:
                texts = df_analyzed[df_analyzed['sentiment'] == sentiment_type]['cleaned_text'].tolist()
            
            if texts:
                wordcloud_fig = create_wordcloud(texts, sentiment_type)
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig)
                else:
                    st.warning("No text data available for word cloud generation.")
        
        with tab4:
            st.subheader("📊 Detailed Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Sentiment Score Statistics:**")
                st.write(df_analyzed['score'].describe())
                
                st.write("**Top Positive Texts:**")
                top_positive = df_analyzed[df_analyzed['sentiment'] == 'Positive'].nlargest(5, 'score')
                for idx, row in top_positive.iterrows():
                    st.write(f"Score: {row['score']:.3f}")
                    # Use the appropriate text column
                    text_to_show = row.get(text_column, row.get('original_text', ''))
                    st.write(f"Text: {str(text_to_show)[:100]}...")
                    st.write("---")
            
            with col2:
                st.write("**Sentiment Distribution:**")
                for sentiment, count in sentiment_counts.items():
                    percentage = (count / len(df_analyzed)) * 100
                    st.write(f"{sentiment}: {count} ({percentage:.1f}%)")
                
                st.write("**Top Negative Texts:**")
                top_negative = df_analyzed[df_analyzed['sentiment'] == 'Negative'].nsmallest(5, 'score')
                for idx, row in top_negative.iterrows():
                    st.write(f"Score: {row['score']:.3f}")
                    # Use the appropriate text column
                    text_to_show = row.get(text_column, row.get('original_text', ''))
                    st.write(f"Text: {str(text_to_show)[:100]}...")
                    st.write("---")
        
        # Real-time update simulation
        if st.button("🔄 Refresh Analysis"):
            st.rerun()
    
    else:
        st.info("👆 Please select a data source from the sidebar to begin analysis.")
        
        # Show sample data structure
        st.subheader("📋 Expected Data Format")
        st.write("Your CSV file should contain at least one text column. Here's an example:")
        
        sample_df = pd.DataFrame({
            'text': [
                "I love this product! It's amazing!",
                "This is the worst experience ever.",
                "It's okay, nothing special.",
                "Absolutely fantastic! Highly recommended!",
                "I hate this so much. Terrible quality."
            ],
            'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='H')
        })
        
        st.dataframe(sample_df, use_container_width=True)

if __name__ == "__main__":
    main()

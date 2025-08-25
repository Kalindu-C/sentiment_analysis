import pandas as pd
import numpy as np
import re
from datetime import datetime
import json
from typing import List, Dict, Optional

# YouTube data extraction
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build

# Text processing
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set up API key (use Colab secrets)
# from google.colab import userdata
import os
from dotenv import load_dotenv
load_dotenv()

YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

# Initialize YouTube API client
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Helper Functions
def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def clean_text(text: str) -> str:
    """Basic text cleaning for YouTube comments"""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?@#]', '', text)
    return text.strip()

def format_duration(seconds: int) -> str:
    """Convert seconds to readable duration"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"

print("✅ All libraries imported and setup complete!")
print(f"✅ YouTube API initialized: {'✓' if YOUTUBE_API_KEY else '✗'}")

# Cell 3: Data Collection Functions

def get_video_transcript(video_id: str) -> Dict:
    """
    Get transcript for a YouTube video
    Returns: dict with transcript text and metadata
    """
    try:
        # Get transcript
        # transcript_list = YouTubeTranscriptApi.fetch(video_id)
        obj = YouTubeTranscriptApi()
        transcript_list = obj.fetch(video_id).to_raw_data()

        # Combine all transcript text
        full_text = ' '.join([entry['text'] for entry in transcript_list])
        
        # Get video metadata
        video_response = youtube.videos().list(
            part='snippet,statistics,contentDetails',
            id=video_id
        ).execute()
        
        if video_response['items']:
            video_info = video_response['items'][0]
            snippet = video_info['snippet']
            stats = video_info['statistics']
            
            return {
                'success': True,
                'video_id': video_id,
                'title': snippet['title'],
                'description': snippet['description'][:500],  # First 500 chars
                'channel': snippet['channelTitle'],
                'published_at': snippet['publishedAt'],
                'view_count': int(stats.get('viewCount', 0)),
                'like_count': int(stats.get('likeCount', 0)),
                'comment_count': int(stats.get('commentCount', 0)),
                'transcript': full_text,
                'transcript_entries': len(transcript_list),
                'duration_seconds': sum([entry['duration'] for entry in transcript_list])
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'video_id': video_id
        }

def get_video_comments(video_id: str, max_comments: int = 100) -> Dict:
    """
    Get comments for a YouTube video
    Returns: dict with comments and metadata
    """
    try:
        comments = []
        next_page_token = None
        
        while len(comments) < max_comments:
            # Calculate how many comments to request in this batch
            batch_size = min(100, max_comments - len(comments))
            
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=batch_size,
                order='relevance',  # Get most relevant comments first
                pageToken=next_page_token
            )
            
            response = request.execute()
            
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'comment_id': item['id'],
                    'author': comment['authorDisplayName'],
                    'text': clean_text(comment['textDisplay']),
                    'like_count': comment['likeCount'],
                    'published_at': comment['publishedAt'],
                    'reply_count': item['snippet']['totalReplyCount']
                })
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        
        return {
            'success': True,
            'video_id': video_id,
            'comments': comments,
            'total_fetched': len(comments),
            'has_more': bool(next_page_token)
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'video_id': video_id,
            'comments': []
        }

def analyze_video(youtube_url: str, max_comments: int = 100) -> Dict:
    """
    Complete analysis pipeline for a YouTube video
    """
    print(f"🎬 Starting analysis for: {youtube_url}")
    
    # Extract video ID
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return {'success': False, 'error': 'Invalid YouTube URL'}
    
    print(f"📝 Video ID extracted: {video_id}")
    
    # Get transcript
    print("📋 Fetching transcript...")
    transcript_data = get_video_transcript(video_id)
    
    if not transcript_data['success']:
        print(f"❌ Transcript error: {transcript_data['error']}")
        return transcript_data
    
    print(f"✅ Transcript fetched: {transcript_data['transcript_entries']} entries")
    print(f"📊 Video: {transcript_data['title'][:50]}...")
    print(f"👀 Views: {transcript_data['view_count']:,}")
    print(f"💬 Comments available: {transcript_data['comment_count']:,}")
    
    # Get comments
    print(f"💬 Fetching up to {max_comments} comments...")
    comments_data = get_video_comments(video_id, max_comments)
    
    if not comments_data['success']:
        print(f"❌ Comments error: {comments_data['error']}")
        return comments_data
    
    print(f"✅ Comments fetched: {comments_data['total_fetched']}")
    
    # Combine results
    return {
        'success': True,
        'video_id': video_id,
        'transcript': transcript_data,
        'comments': comments_data,
        'analysis_timestamp': datetime.now().isoformat()
    }


# Test with a sample video
print("\n" + "="*50)
print("🧪 TESTING DATA COLLECTION")
print("="*50)

# Example URLs to test (replace with your preferred video)
test_urls = [
    "https://www.youtube.com/watch?v=eiNVjkJDsqI",  # Famous video with many comments
    # "https://youtu.be/dQw4w9WgXcQ"  # Short format
]

print("Example usage:")
print("result = analyze_video('YOUR_YOUTUBE_URL_HERE', max_comments=50)")
print("\nReplace with actual video URL to test!")


if __name__ == "__main__":
    # Uncomment below to test with actual URL
    test_url = "https://www.youtube.com/watch?v=eiNVjkJDsqI"
    result = analyze_video(test_url, max_comments=20)
    print(json.dumps(result, indent=2, default=str))
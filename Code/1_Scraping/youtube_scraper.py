import csv
import time
import os
import re
import pandas as pd
from googleapiclient.discovery import build

# Configure paths
PROJECT_DIR = "~/Desktop/NLP_Submission/Data"
SAVE_FOLDER = os.path.join(PROJECT_DIR, "/1_scraped_data/scraped_youtube")
os.makedirs(SAVE_FOLDER, exist_ok=True)

# YouTube API setup
API_KEY = "XXXXXXXXX"
youtube = build('youtube', 'v3', developerKey=API_KEY)

def extract_channel_id(url):
    if pd.isna(url) or url.strip() in ('--', '', ' '):
        return None

    # Clean URL parameters
    clean_url = url.split('?')[0]

    # Handle channel ID URLs
    if '/channel/' in clean_url:
        match = re.search(r'/channel/([\w-]+)', clean_url)
        return match.group(1) if match else None

    # Handle custom URLs
    if '/@' in clean_url:
        handle = clean_url.split('/@')[1].split('/')[0]
        try:
            request = youtube.channels().list(
                part="id",
                forHandle=f"@{handle}"
            )
            response = request.execute()
            if response.get('items'):
                return response['items'][0]['id']
        except Exception as e:
            print(f"Error resolving @{handle}: {str(e)}")
            return None

    # Handle alternative URL formats
    if 'youtube.com/user/' in clean_url:
        username = clean_url.split('/user/')[1].split('/')[0]
        try:
            request = youtube.channels().list(
                part="id",
                forUsername=username
            )
            response = request.execute()
            if response.get('items'):
                return response['items'][0]['id']
        except Exception as e:
            print(f"Error resolving username {username}: {str(e)}")
            return None

    print(f"Unsupported URL format: {url}")
    return None

def get_channel_videos(channel_id):
    videos = []
    next_page_token = None
    
    while True:
        try:
            request = youtube.search().list(
                part="id,snippet",
                channelId=channel_id,
                maxResults=50,
                order="date",
                pageToken=next_page_token,
                type="video"
            )
            response = request.execute()
            videos.extend(response.get('items', []))
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        except Exception as e:
            print(f"Video fetch error: {str(e)}")
            break
    
    return [item['id']['videoId'] for item in videos if 'videoId' in item['id']]

def get_video_comments(video_id, max_comments=None):
    comments = []
    next_page_token = None
    
    try:
        while True:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                textFormat="plainText",
                pageToken=next_page_token
            )
            response = request.execute()
            
            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'author': comment['authorDisplayName'],
                    'text': comment['textDisplay'],
                    'date': comment['publishedAt'].split('T')[0]
                })
                if max_comments and len(comments) >= max_comments:
                    return comments
            
            next_page_token = response.get('nextPageToken')
            if (max_comments and len(comments) >= max_comments) or not next_page_token:
                break
    except Exception as e:
        print(f"Comment error {video_id}: {str(e)}")
    
    return comments

def main():
    input_csv = os.path.join(PROJECT_DIR, 'all_streamer.csv')
    output_csv = os.path.join(SAVE_FOLDER, 'scraped_youtube.csv')
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=['channel_url', 'video_id', 'author', 'text', 'date', 'nick'])
        writer.writeheader()

        with open(input_csv, 'r', encoding='utf-8') as in_file:
            reader = csv.DictReader(in_file)
            
            for row in reader:
                channel_url = row.get('YouTube Link', '').strip()
                nick = row.get('Nick', '').strip()
                
                if not channel_url:
                    print(f"\n⚠️ Missing URL for: {nick}")
                    continue
                
                channel_id = extract_channel_id(channel_url)
                if not channel_id:
                    print(f"\n⚠️ Could not extract ID from: {channel_url} (Nick: {nick})")
                    continue
                
                print(f"\nProcessing {nick} ({channel_url})...")
                video_ids = get_channel_videos(channel_id)
                
                total_comments = 0
                for vid in video_ids:
                    if total_comments >= 10000:
                        break
                    remaining = 10000 - total_comments
                    comments = get_video_comments(vid, max_comments=remaining)
                    
                    for comment in comments:
                        writer.writerow({
                            **comment,
                            'channel_url': channel_url,
                            'video_id': vid,
                            'nick': nick
                        })
                    
                    total_comments += len(comments)
                    print(f"Collected {len(comments)} comments (Total: {total_comments})")
                    time.sleep(1.5)

if __name__ == "__main__":
    main()
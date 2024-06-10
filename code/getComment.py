import praw
import json
from tqdm import tqdm
import os

# Initialize PRAW with your credentials
reddit = praw.Reddit(client_id='TefhPe0MYpOz3DrGNiCsRg',
                     client_secret='lK7E-NZjpVo5YHDwCpZibUVlPuQnKA',
                     user_agent='Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0')

# Prompt user to input subreddit name
subreddit_name = input("Enter subreddit name: ")

subreddit = reddit.subreddit(subreddit_name)

comments_list = []
num_comments = 700  # Number of comments to collect

# Use tqdm to create a progress bar
with tqdm(total=num_comments, desc="Collecting Comments") as pbar:
    for i, comment in enumerate(subreddit.comments(limit=num_comments)):
        sentences = comment.body.split('. ')
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                comments_list.append({
                    'subreddit': comment.subreddit.display_name,
                    'sentence': sentence
                })
        pbar.update(1)  # Update progress bar after each comment is collected
        if i >= num_comments - 1:
            break

filename = f"./JSON/prefer/{subreddit_name}.json"

# Open the file with 'w' mode if it exists, otherwise with 'x' mode
mode = 'w' if os.path.exists(filename) else 'x'
with open(filename, mode, encoding='utf-8') as f:
    json.dump(comments_list, f, ensure_ascii=False, indent=4)

print(f"File {filename} created and comments have been saved. Total comments: {len(comments_list)}")

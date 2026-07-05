import kagglehub
import os

# Download WSB posts
print("Downloading r/wallstreetbets dataset...")
path1 = kagglehub.dataset_download("unanimad/reddit-rwallstreetbets")
print(f"Downloaded to: {path1}")

# Download multi-subreddit dataset (includes r/investing, r/stocks)
print("Downloading multi-subreddit dataset...")
path2 = kagglehub.dataset_download("shergreen/wallstreetbets-subreddit-submissions")
print(f"Downloaded to: {path2}")

print("Done. Check the paths above for your CSV files.")

import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


class SpiceRatingVideoDataset(Dataset):
    def __init__(self, csv_path='spice_ratings.csv', video_folder='data', frames_per_clip=16, width=192, height=108):
        self.df = pd.read_csv(csv_path)
        self.video_folder = video_folder
        self.frames_per_clip = frames_per_clip
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get clip_number and label
        row = self.df.iloc[idx]
        clip_number = row['clip_number']
        label = float(row['spice_rating'])  # 0-10 rating float

        video_path = os.path.join(self.video_folder, f"{clip_number}.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found")

        # Load video frames
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # We want to sample frames_per_clip (=16) frames evenly from the clip
        # If less frames, we will loop or pad with last frame
        indices = np.linspace(0, frame_count - 1, num=self.frames_per_clip, dtype=int)

        frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                # If read fails, repeat last frame or fill with zeros
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((self.height, self.width), dtype=np.uint8))
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Resize to (width, height)
            resized = cv2.resize(gray, (self.width, self.height))
            frames.append(resized)
        cap.release()

        # Convert list of frames to numpy array shape (frames, height, width)
        frames_np = np.stack(frames, axis=0)  # (16, 108, 192)

        # Add channel dim -> (frames, 1, height, width)
        frames_np = np.expand_dims(frames_np, axis=1)

        # Convert to torch tensor, float, normalize to [0,1]
        frames_tensor = torch.from_numpy(frames_np).float() / 255.0

        # The model expects input shape: (batch, frames, channels, width, height)
        # so for one sample: (frames, channels, width, height)
        # but width and height in tensor are (height, width), so we permute last two dims
        frames_tensor = frames_tensor.permute(0, 1, 3, 2)  # (16, 1, 192, 108)

        return frames_tensor, torch.tensor(label, dtype=torch.float32)


if __name__ == "__main__":
    dataset = SpiceRatingVideoDataset(csv_path='data/spice_ratings.csv', video_folder='C:\\Users\giant\PycharmProjects\AddSpice\data\\videos')

    # Load first sample
    video_tensor, rating = dataset[30]  # video_tensor shape: (16, 1, 192, 108)

    # Extract first frame and remove channel dim: (1, 192, 108) -> (192, 108)
    first_frame = video_tensor[0].squeeze(0).numpy()

    plt.imshow(first_frame, cmap='gray')
    plt.title(f'First frame, rating: {rating.item():.2f}')
    plt.axis('off')
    plt.show()
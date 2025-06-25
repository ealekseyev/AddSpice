import torch
import sys
import cv2
import numpy as np
from architecture import VideoRatingModel  # your model file


def load_video(video_path, frames_per_clip=16, width=192, height=108):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, frame_count - 1, num=frames_per_clip, dtype=int)

    frames = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((height, width), dtype=np.uint8))
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (width, height))
        frames.append(resized)
    cap.release()

    frames_np = np.stack(frames, axis=0)  # (frames, height, width)
    frames_np = np.expand_dims(frames_np, axis=1)  # (frames, 1, height, width)
    frames_tensor = torch.from_numpy(frames_np).float() / 255.0
    frames_tensor = frames_tensor.permute(0, 1, 3, 2)  # (frames, channels, width, height)
    return frames_tensor.unsqueeze(0)  # add batch dim (1, frames, channels, width, height)


def main(idx):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VideoRatingModel().to(device)
    model = torch.load('spicy_model_e9.pth', map_location=device, weights_only=False)
    model.eval()

    video_path = f"data/videos/{idx}.mp4"
    video_tensor = load_video(video_path).to(device)

    with torch.no_grad():
        output = model(video_tensor)  # output shape: (1,1)

    rating = output.item()
    print(f"Predicted spice rating for video {idx}: {rating:.2f} (0-10 scale)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <video_number>")
        sys.exit(1)
    video_idx = sys.argv[1]
    main(video_idx)

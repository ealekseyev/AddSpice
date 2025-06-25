import cv2
import csv
import os
import time
import sys
import numpy as np
import torch
from architecture import VideoRatingModel  # your model file

# Configuration parameters - set these at the top
INPUT_VIDEO = "C:/Users/giant/Downloads/ONE HOUR of Mountain Biking POV (GoPro) Footage.mp4"
CSV_OUTPUT_PATH = "C:/Users/giant/PycharmProjects/AddSpice/data/spice_ratings.csv"
CLIP_DURATION = 1.0  # seconds per clip (should match the clip generation)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VideoRatingModel().to(device)
model = torch.load('spicy_model_e9.pth', map_location=device, weights_only=False)
model.eval()

def getSpiciness(tensor):
    with torch.no_grad():
        output = model(torch.tensor(tensor, dtype=torch.float32).to(device))  # output shape: (1,1)
    rating = output.item()
    return rating


def setup_csv():
    """Setup CSV file and return the starting clip number"""
    headers = ['clip_number', 'spice_rating']

    if os.path.exists(CSV_OUTPUT_PATH):
        # CSV exists, find the last entry to resume from
        try:
            with open(CSV_OUTPUT_PATH, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                last_clip = -1
                for row in reader:
                    if row['clip_number'].strip():  # Skip empty rows
                        last_clip = int(row['clip_number'])

                start_clip = last_clip + 1
                print(f"Resuming from clip {start_clip} (CSV has {last_clip + 1} entries)")
                return start_clip

        except (ValueError, KeyError) as e:
            print(f"Error reading existing CSV: {e}")
            print("Starting from the beginning...")
            return 0
    else:
        # Create new CSV file
        os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)
        with open(CSV_OUTPUT_PATH, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
        print(f"Created new CSV file: {CSV_OUTPUT_PATH}")
        return 0


def write_to_csv(clip_number, spice_rating):
    """Write a single entry to the CSV file"""
    try:
        with open(CSV_OUTPUT_PATH, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['clip_number', 'spice_rating'])
            writer.writerow({'clip_number': clip_number, 'spice_rating': spice_rating})
        print(f"Saved: Clip {clip_number} = Rating {spice_rating}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")


def get_video_info(video_path):
    """Get video information"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    cap.release()
    return fps, total_frames, duration

def prepare_tensor_from_frames(frames):
    processed_frames = []

    for frame in frames:
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (192, 108))  # width x height
        processed_frames.append(frame)

    if len(processed_frames) < 16:
        last = processed_frames[-1]
        processed_frames += [last] * (16 - len(processed_frames))
    else:
        processed_frames = processed_frames[:16]

    # (frames, height, width)
    tensor = np.stack(processed_frames)  # shape: (16, 108, 192)

    # Add channel dim: (frames, channels=1, height, width)
    tensor = np.expand_dims(tensor, axis=1)  # (16, 1, 108, 192)

    # Convert to torch tensor and normalize
    tensor = torch.tensor(tensor, dtype=torch.float32) / 255.0  # still 4D

    # Add batch dimension: (batch=1, frames, channels, height, width)
    tensor = tensor.unsqueeze(0)  # now 5D: (1, 16, 1, 108, 192)

    # Permute to (batch, frames, channels, width, height) as model expects
    tensor = tensor.permute(0, 1, 2, 4, 3)  # (1, 16, 1, 192, 108)

    return tensor.to(device)



def label_video():
    """Main video labeling function"""
    print("Video Clip Labeling Tool")
    print("=" * 40)
    print(f"Input video: {INPUT_VIDEO}")
    print(f"Output CSV: {CSV_OUTPUT_PATH}")
    print("Instructions:")
    print("- Watch each 1-second clip")
    print("- Press number keys (0-9) to rate the clip")
    print("- Press 'q' to quit and save progress")
    print("- Press 's' to skip current clip without rating")
    print("- Spiciness score will be shown in top-right corner")
    print("- You can resume labeling later from where you left off")
    print("=" * 40)

    # Check if input video exists
    if not os.path.exists(INPUT_VIDEO):
        print(f"Error: Input video '{INPUT_VIDEO}' not found!")
        return

    # Get video information
    fps, total_frames, duration = get_video_info(INPUT_VIDEO)
    if fps is None:
        print("Error: Could not read video information!")
        return

    print(f"Video info: {duration:.1f}s, {fps:.1f}fps, {total_frames} frames")

    # Setup CSV and get starting position
    start_clip = setup_csv()
    total_clips = int(duration / CLIP_DURATION)

    if start_clip >= total_clips:
        print("All clips have already been labeled!")
        return

    print(f"Will label clips {start_clip} to {total_clips - 1}")
    print("\nPress any key to start...")
    input()

    # Open video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("Error: Could not open video!")
        return

    # Skip to the starting position
    start_frame = int(start_clip * CLIP_DURATION * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_clip = start_clip

    print(f"\nStarting with clip {current_clip}")
    print("Controls: 0-9 to rate, 'q' to quit, 's' to skip")

    while current_clip < total_clips:
        # Set position to exact start of current clip
        clip_start_frame = int(current_clip * CLIP_DURATION * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame)

        print(f"\nClip {current_clip}/{total_clips - 1} - Preparing clip...")

        # Collect frames for the entire 1-second clip first
        frames_to_play = int(fps * CLIP_DURATION)
        clip_frames = []

        for frame_count in range(frames_to_play):
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            clip_frames.append(frame.copy())

        if not clip_frames:
            break

        # Prepare tensor and get spiciness score
        print(f"Calculating spiciness for clip {current_clip}...")
        tensor = prepare_tensor_from_frames(clip_frames)
        spiciness_score = getSpiciness(tensor)

        print(f"Playing clip {current_clip} (Spiciness: {spiciness_score})...")

        # Now play the clip with spiciness overlay
        clip_start_time = time.time()

        for frame_count, frame in enumerate(clip_frames):
            # Resize frame if too large (for performance)
            height, width = frame.shape[:2]
            if width > 1280:
                scale = 1280 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))

            # Add spiciness score overlay in top-right corner
            display_frame = frame.copy()
            text = f"Spiciness: {spiciness_score:.2f}"

            # Get text size for positioning
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

            # Position in top-right corner with some padding
            text_x = display_frame.shape[1] - text_width - 10
            text_y = text_height + 10

            # Add background rectangle for better visibility
            cv2.rectangle(display_frame,
                          (text_x - 5, text_y - text_height - 5),
                          (text_x + text_width + 5, text_y + baseline + 5),
                          (0, 0, 0), -1)

            # Add text
            cv2.putText(display_frame, text, (text_x, text_y),
                        font, font_scale, (0, 255, 255), font_thickness)

            # Display the frame
            cv2.imshow('Video Labeling - Rate this clip (0-9) or q to quit', display_frame)

            # Check for immediate key press (to skip to next clip faster)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # A key was pressed
                if key == ord('q'):
                    print("\nQuitting and saving progress...")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key == ord('s'):
                    print("Skipping clip...")
                    break
                elif key >= ord('0') and key <= ord('9'):
                    rating = key - ord('0')
                    write_to_csv(current_clip, rating)
                    break

            # Control playback speed
            elapsed = time.time() - clip_start_time
            expected_time = frame_count / fps
            if elapsed < expected_time:
                time.sleep(expected_time - elapsed)

        # If we completed the clip without input, wait for rating
        if frame_count >= len(clip_frames) - 1:
            print(f"Clip {current_clip} finished. Enter rating (0-9), 's' to skip, or 'q' to quit:")

            # Wait for user input
            while True:
                key = cv2.waitKey(0) & 0xFF

                if key == ord('q'):
                    print("\nQuitting and saving progress...")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key == ord('s'):
                    print("Skipping clip...")
                    break
                elif key >= ord('0') and key <= ord('9'):
                    rating = key - ord('0')
                    write_to_csv(current_clip, rating)
                    break
                else:
                    print("Invalid key. Press 0-9 to rate, 's' to skip, or 'q' to quit.")

        # Move to next clip
        current_clip += 1

        # Check if window was closed
        if cv2.getWindowProperty('Video Labeling - Rate this clip (0-9) or q to quit', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    print(f"\nLabeling session completed!")
    print(f"Progress saved to: {CSV_OUTPUT_PATH}")
    print(f"Labeled clips: {start_clip} to {current_clip - 1}")


def main():
    """Main function"""
    try:
        label_video()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
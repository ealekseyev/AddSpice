import cv2
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configuration parameters - set these at the top
INPUT_VIDEOS = [
    "C:\\Users\giant\Downloads\\videoplayback.mp4"
]

OUTPUT_FOLDER = "C:/Users/giant/PycharmProjects/AddSpice/data/redbull1"
TARGET_WIDTH = 192
TARGET_HEIGHT = 108
TARGET_FPS = 16.0
CONVERT_GRAYSCALE = True
CLIP_DURATION = 1.0  # seconds per clip

# Global counter for sequential clip numbering
clip_counter = 0
counter_lock = threading.Lock()


def get_next_clip_number():
    """Thread-safe way to get the next clip number"""
    global clip_counter
    with counter_lock:
        current = clip_counter
        clip_counter += 1
        return current


def create_output_folder():
    """Create output folder if it doesn't exist"""
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")
    else:
        print(f"Using existing output folder: {OUTPUT_FOLDER}")


def process_single_video(video_path, video_index):
    """Process a single video into 1-second clips"""
    print(f"[Video {video_index}] Starting processing: {video_path}")

    if not os.path.exists(video_path):
        print(f"[Video {video_index}] Error: File '{video_path}' not found!")
        return []

    # Open input video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[Video {video_index}] Error: Could not open video file")
        return []

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_duration = total_frames / original_fps

    print(
        f"[Video {video_index}] Properties: {original_width}x{original_height}, {original_fps:.2f}fps, {original_duration:.2f}s")

    # Calculate frames per clip
    frames_per_clip = int(TARGET_FPS * CLIP_DURATION)
    total_clips = int(original_duration / CLIP_DURATION)

    print(f"[Video {video_index}] Will create {total_clips} clips of {frames_per_clip} frames each")

    # Process clips
    clip_files = []
    current_clip = 0
    frames_in_current_clip = 0
    current_clip_number = None
    out = None

    # Frame processing variables
    frame_count = 0
    processed_frames = 0

    if TARGET_FPS <= original_fps:
        # Downsampling
        frame_step = original_fps / TARGET_FPS
        next_frame_to_keep = 0.0
    else:
        # Upsampling
        frame_repeat_ratio = TARGET_FPS / original_fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        should_process_frame = False
        frame_repeats = 1

        if TARGET_FPS <= original_fps:
            # Downsampling logic
            if frame_count >= int(next_frame_to_keep):
                should_process_frame = True
                next_frame_to_keep += frame_step
        else:
            # Upsampling logic
            should_process_frame = True
            repeats = int(frame_repeat_ratio)
            fractional_part = frame_repeat_ratio - repeats
            if (frame_count * fractional_part) % 1.0 >= 0.5:
                repeats += 1
            frame_repeats = repeats

        if should_process_frame:
            # Resize frame
            resized_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

            # Convert to grayscale if requested
            if CONVERT_GRAYSCALE:
                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            # Process frame repeats
            for _ in range(frame_repeats):
                # Start new clip if needed
                if frames_in_current_clip == 0:
                    current_clip_number = get_next_clip_number()
                    clip_filename = os.path.join(OUTPUT_FOLDER, f"{current_clip_number}.mp4")

                    # Create VideoWriter for new clip
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(clip_filename, fourcc, TARGET_FPS,
                                          (TARGET_WIDTH, TARGET_HEIGHT),
                                          isColor=not CONVERT_GRAYSCALE)

                    if not out.isOpened():
                        print(f"[Video {video_index}] Error: Could not create clip {current_clip_number}")
                        break

                # Write frame to current clip
                out.write(resized_frame)
                frames_in_current_clip += 1
                processed_frames += 1

                # Check if clip is complete
                if frames_in_current_clip >= frames_per_clip:
                    # Finish current clip
                    out.release()
                    clip_files.append(f"{current_clip_number}.mp4")
                    print(f"[Video {video_index}] Created clip: {current_clip_number}.mp4")

                    # Reset for next clip
                    frames_in_current_clip = 0
                    current_clip += 1
                    out = None

        frame_count += 1

        # Show progress
        if frame_count % 200 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"[Video {video_index}] Progress: {progress:.1f}% ({current_clip + 1} clips created)")

    # Handle any remaining frames in the last clip
    if out is not None and frames_in_current_clip > 0:
        out.release()
        clip_files.append(f"{current_clip_number}.mp4")
        print(f"[Video {video_index}] Created final clip: {current_clip_number}.mp4 ({frames_in_current_clip} frames)")

    cap.release()

    print(f"[Video {video_index}] Completed! Created {len(clip_files)} clips")
    return clip_files


def process_videos_parallel():
    """Process all videos in parallel"""
    print("Starting parallel video processing...")
    print(f"Configuration:")
    print(f"- Target resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"- Target framerate: {TARGET_FPS} fps")
    print(f"- Clip duration: {CLIP_DURATION} seconds")
    print(f"- Grayscale: {'Yes' if CONVERT_GRAYSCALE else 'No'}")
    print(f"- Output folder: {OUTPUT_FOLDER}")
    print(f"- Input videos: {len(INPUT_VIDEOS)}")

    # Create output folder
    create_output_folder()

    # Process videos in parallel
    all_clips = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=min(len(INPUT_VIDEOS), 4)) as executor:
        # Submit all video processing tasks
        future_to_video = {
            executor.submit(process_single_video, video_path, i): (video_path, i)
            for i, video_path in enumerate(INPUT_VIDEOS)
        }

        # Collect results as they complete
        for future in as_completed(future_to_video):
            video_path, video_index = future_to_video[future]
            try:
                clips = future.result()
                all_clips.extend(clips)
                print(f"[Video {video_index}] Finished processing {video_path}")
            except Exception as e:
                print(f"[Video {video_index}] Error processing {video_path}: {str(e)}")

    end_time = time.time()

    print(f"\nProcessing completed!")
    print(f"Total clips created: {len(all_clips)}")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print(f"Clips saved in: {OUTPUT_FOLDER}")

    # Show clip numbering summary
    print(f"\nClip numbering:")
    print(f"- Clips are numbered sequentially: 0.mp4, 1.mp4, 2.mp4, ...")
    print(f"- Each video's clips continue from where the previous video ended")
    print(f"- Final clip number: {clip_counter - 1}.mp4")


def main():
    """Main function"""
    try:
        # Validate input videos exist
        missing_videos = [video for video in INPUT_VIDEOS if not os.path.exists(video)]
        if missing_videos:
            print("Error: The following video files were not found:")
            for video in missing_videos:
                print(f"  - {video}")
            print("Please check the file paths in INPUT_VIDEOS")
            return

        # Process all videos
        process_videos_parallel()

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")


if __name__ == "__main__":
    main()
import cv2
import os


def get_user_inputs():
    """Get all required parameters from user input"""
    print("Video Processing Tool")
    print("=" * 30)

    # Get input file path
    input_file = input("Enter the path to your MP4 file: ").strip()

    # Validate input file exists
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found!")
        return None

    # Get target resolution
    print("\nEnter target resolution:")
    try:
        width = int(input("Width (pixels): "))
        height = int(input("Height (pixels): "))
    except ValueError:
        print("Error: Please enter valid numbers for resolution!")
        return None

    # Get target framerate
    try:
        target_fps = float(input("Target framerate (fps): "))
    except ValueError:
        print("Error: Please enter a valid number for framerate!")
        return None

    # Ask about grayscale conversion
    grayscale_choice = input("Convert to grayscale? (y/n): ").strip().lower()
    convert_grayscale = grayscale_choice in ['y', 'yes']

    return {
        'input_file': input_file,
        'width': width,
        'height': height,
        'target_fps': target_fps,
        'convert_grayscale': convert_grayscale
    }


def process_video(params):
    """Process the video with the given parameters"""
    input_file = params['input_file']
    width = params['width']
    height = params['height']
    target_fps = params['target_fps']
    convert_grayscale = params['convert_grayscale']

    # Generate output filename
    base_name = os.path.splitext(input_file)[0]
    grayscale_suffix = "_grayscale" if convert_grayscale else ""
    output_file = f"{base_name}_processed_{width}x{height}_{target_fps}fps{grayscale_suffix}.mp4"

    # Open input video
    cap = cv2.VideoCapture(input_file)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_file}'")
        return False

    # Get original video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nOriginal video properties:")
    print(f"Resolution: {original_width}x{original_height}")
    print(f"Framerate: {original_fps:.2f} fps")
    print(f"Total frames: {total_frames}")

    print(f"\nTarget properties:")
    print(f"Resolution: {width}x{height}")
    print(f"Framerate: {target_fps} fps")
    print(f"Grayscale: {'Yes' if convert_grayscale else 'No'}")
    print(f"Output file: {output_file}")

    # Define codec and create VideoWriter
    # Output at target framerate - this determines playback speed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, target_fps, (width, height),
                          isColor=not convert_grayscale)

    if not out.isOpened():
        print("Error: Could not create output video file")
        cap.release()
        return False

    # For proper framerate conversion that maintains playback duration:
    # Process frames sequentially for better performance

    original_duration = total_frames / original_fps
    target_total_frames = int(original_duration * target_fps)

    print(f"Original duration: {original_duration:.2f} seconds")
    print(f"Target total frames needed: {target_total_frames}")

    frame_count = 0
    processed_frames = 0

    print("\nProcessing video...")

    if target_fps <= original_fps:
        # Downsampling: skip frames
        frame_step = original_fps / target_fps
        next_frame_to_keep = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Check if we should keep this frame
            if frame_count >= int(next_frame_to_keep):
                # Resize frame
                resized_frame = cv2.resize(frame, (width, height))

                # Convert to grayscale if requested
                if convert_grayscale:
                    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

                # Write frame to output
                out.write(resized_frame)
                processed_frames += 1

                # Calculate next frame to keep
                next_frame_to_keep += frame_step

            frame_count += 1

            # Show progress
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({processed_frames} frames written)")

    else:
        # Upsampling: duplicate frames
        frame_repeat_ratio = target_fps / original_fps

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame
            resized_frame = cv2.resize(frame, (width, height))

            # Convert to grayscale if requested
            if convert_grayscale:
                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            # Calculate how many times to repeat this frame
            repeats = int(frame_repeat_ratio)

            # Handle fractional part for more accurate timing
            fractional_part = frame_repeat_ratio - repeats
            if (frame_count * fractional_part) % 1.0 >= 0.5:
                repeats += 1

            # Write frame multiple times
            for _ in range(repeats):
                out.write(resized_frame)
                processed_frames += 1

            frame_count += 1

            # Show progress
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({processed_frames} frames written)")

    print(f"Final: {processed_frames} frames written")

    # Release resources
    cap.release()
    out.release()

    print(f"\nVideo processing completed!")
    print(f"Output saved as: {output_file}")
    print(f"Processed {processed_frames} frames")

    return True


def main():
    """Main function to run the video processing tool"""
    try:
        # Get user inputs
        params = get_user_inputs()

        if params is None:
            print("Exiting due to invalid input.")
            return

        # Process the video
        success = process_video(params)

        if success:
            print("\nVideo processing completed successfully!")
        else:
            print("\nVideo processing failed!")

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")


if __name__ == "__main__":
    main()

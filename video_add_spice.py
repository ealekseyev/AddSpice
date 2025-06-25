import cv2
import torch
import numpy as np
from architecture import VideoRatingModel


def preprocess_clip(frames, width=192, height=108):
    """Convert list of BGR frames to model input tensor: (1, 16, 1, 192, 108)"""
    processed = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (width, height))
        processed.append(resized)

    clip_np = np.stack(processed, axis=0)  # (16, 108, 192)
    clip_np = np.expand_dims(clip_np, axis=1)  # (16, 1, 108, 192)
    clip_tensor = torch.from_numpy(clip_np).float() / 255.0
    clip_tensor = clip_tensor.permute(0, 1, 3, 2)  # (16, 1, 192, 108)
    return clip_tensor.unsqueeze(0)  # (1, 16, 1, 192, 108)


def main(video_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VideoRatingModel().to(device)
    model = torch.load('spicy_model_e9.pth', map_location=device, weights_only=False)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Playing video: {video_path} ({fps} fps)")

    frame_buffer = []
    frame_count = 0
    rating = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame)
        frame_count += 1

        if len(frame_buffer) == 16:
            # Preprocess and predict
            clip_tensor = preprocess_clip(frame_buffer).to(device)
            with torch.no_grad():
                pred = model(clip_tensor)
                rating = pred.item()

            # Draw rating on each frame in buffer and play them
            for f in frame_buffer:
                display = f.copy()
                text = f"Spice: {rating:.2f}/10"
                cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Spicy Video", display)

                if cv2.waitKey(int(1000 // fps)) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            frame_buffer = []  # reset for next second

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python spice_overlay_player.py <video_file>")
        sys.exit(1)
    main(sys.argv[1])

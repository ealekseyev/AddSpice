AddSpice: Real-Time Spice Rating from Mountain Bike GoPro Footage
AddSpiec is a deep learning project that uses a 3D convolutional neural network (CNN) to rate the "spiciness" of GoPro footage captured during mountain biking. The model processes live video input and outputs a Spice Rating from 0 to 10, where:

0 = stationary (chill)

10 = Red Bull Rampage level insanity

This system is built as a gag for mountain bikers to rate their footage and compete for the craziest rides!

🚴‍♂️ How It Works
Input: Live video feed from a GoPro mounted on a mountain bike.

Preprocessing:

Convert to grayscale for performance.

Resize each frame to 192x108 pixels.

Downsample the feed to 16 FPS.

Slice the video into 1-second chunks (16 frames per chunk).

Model:

A PyTorch-based 3D CNN trained on annotated mountain biking clips.

Accepts input shape: (1, 1, 16, 192, 108) → (batch, channels, frames, height, width).

Output: A floating-point Spice Rating from 0.0 to 10.0 displayed in real time on the video (top-left corner).

🧠 Model Architecture
Input: 1-second video clip (16 grayscale frames)

Layers:

3D Conv layers

ReLU activations

MaxPooling

Repeat above thrice

Fully connected output layer with scalar regression output

Loss: Mean Squared Error (MSE)

Optimizer: Adam


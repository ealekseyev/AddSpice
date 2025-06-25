import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from video_dataset import SpiceRatingVideoDataset  # your dataset class file
from architecture import VideoRatingModel  # your model class file


def train():
    # Hyperparameters
    batch_size = 4
    learning_rate = 1e-4
    epochs = 10

    # Device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Dataset and DataLoader
    dataset = SpiceRatingVideoDataset(csv_path='data/spice_ratings.csv',
                                      video_folder='C:\\Users\giant\PycharmProjects\AddSpice\data\\videos')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Model, loss, optimizer
    model = VideoRatingModel().to(device)
    criterion = nn.MSELoss()  # regression loss for continuous values
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)  # (batch, frames, channels, width, height)
            labels = labels.to(device).unsqueeze(1)  # shape (batch, 1)

            optimizer.zero_grad()
            outputs = model(inputs)  # outputs in [0,10], shape (batch,1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 10 == 9:  # print every 10 batches
                avg_loss = running_loss / 10
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {avg_loss:.4f}")
                running_loss = 0.0
        torch.save(model, f'spicy_model_e{epoch}.pth')

    print("Training finished.")


if __name__ == "__main__":
    train()

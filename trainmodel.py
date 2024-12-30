import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from ConvLSTM3D_model.Seq2Seq import Seq2Seq
from torch.utils.data import DataLoader
#import torch.nn.functional as f
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# import io
# import imageio
# from ipywidgets import widgets, HBox

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Torch.nn version: {torch.__version__}")
print(f"matplotlib version : {plt.matplotlib.__version__}")
if Seq2Seq is not None:
    print("Custom Seq2Seq model imported successfully.")
else:
    print("Error importing Seq2Seq.")


class Normalize:
    def __init__(self, channels):
        self.channels = channels
        self.minimums = []
        self.maximums = []

    def normalize(self, data):
        self.minimums = []
        self.maximums = []
        data = np.array(data).swapaxes(0, 1)
        for i in range(0, self.channels):
            # print(np.min(channel), np.max(channel))
            min = np.min(data[i])
            max = np.max(data[i])

            if max - min == 0:
                self.maximums.append(1)
                self.minimums.append(0)

                if max != 0:
                    data[i] = np.divide(data[i], max)


            else:

                data[i] = (data[i] - min) / (max - min)

                self.minimums.append(min)
                self.maximums.append(max)

            print(np.min(data[i]), np.max(data[i]))

        data = data.swapaxes(0, 1)
        print(np.shape(data))
        return data, self.minimums, self.maximums

    def normalize_other(self, data):
        data = np.array(data).swapaxes(0, 1)
        for i in range(0, self.channels):
            data[i] = (data[i] - self.minimums[i]) / (self.maximums[i] - self.minimums[i])

        data = data.swapaxes(0, 1)

        return data
    # New denormalize method
    def denormalize(self, data):
        """
        Denormalizes the normalized data back to its original scale using the stored minimums and maximums.
        """
        data = np.array(data).swapaxes(0, 1)
        for i in range(0, self.channels):
            data[i] = data[i] * (self.maximums[i] - self.minimums[i]) + self.minimums[i]
        data = data.swapaxes(0, 1)
        return data

def save_comparison_plot(first_frame, second_frame, third_frame, predicted_frame=None, index=0, depth=0,file = ""):
    """
    Saves a comparison plot of the first frame, second frame, third frame, and optionally, the predicted frame.

    Args:
        first_frame: The first frame (denormalized).
        second_frame: The second frame (denormalized).
        third_frame: The actual third frame (denormalized).
        predicted_frame: The predicted third frame (optional, denormalized).
        index: Index of the current plot (used in the filename).
        depth: Depth slice of the 3D data to visualize.
    """
    plt.figure(figsize=(20, 5))

    # Plot the first frame
    plt.subplot(1, 4 if predicted_frame is not None else 3, 1)
    plt.imshow(first_frame[depth], cmap='viridis')
    plt.title("First Frame")

    # Plot the second frame
    plt.subplot(1, 4 if predicted_frame is not None else 3, 2)
    plt.imshow(second_frame[depth], cmap='viridis')
    plt.title("Second Frame")

    # Plot the actual third frame
    plt.subplot(1, 4 if predicted_frame is not None else 3, 3)
    plt.imshow(third_frame[depth], cmap='viridis')
    plt.title("Third Frame")

    # Plot the predicted frame if provided
    if predicted_frame is not None:
        plt.subplot(1, 4, 4)
        plt.imshow(predicted_frame[depth], cmap='viridis')
        plt.title("Predicted Frame")

    # Save the plot
    file_name = f"{file}_comparison_plot_{index + 1}.png"
    plt.savefig(file_name)
    plt.close()
    print(f"Saved plot: {file_name}")


def calculate_metrics(actual_frame, predicted_frame):
    """
    Calculates and prints the RMSE and MAE between the actual and predicted frames.

    Args:
        actual_frame (numpy.ndarray): The actual frame.
        predicted_frame (numpy.ndarray): The predicted frame.

    Returns:
        tuple: RMSE and MAE values.
    """
    # Flatten the frames to ensure 1D comparison
    actual_flat = actual_frame.flatten()
    predicted_flat = predicted_frame.flatten()

    # Calculate MAE
    mae = mean_absolute_error(actual_flat, predicted_flat)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actual_flat, predicted_flat))

    # Print metrics
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")

    return rmse, mae
# seed = 42
# np.random.seed(seed)
# random.seed(seed)
# torch.manual_seed(seed)

data=np.float32(np.load(r"D:\files2\output.npy")[:, :, :, 0:10, 100:132, 200:232])
np.random.shuffle(data)
train_data = data[:2]
val_data =  data[2:]
print(np.shape(data))
print(np.shape(train_data))
print(np.shape(val_data))
print(train_data.dtype)


train_data = data[:3]
val_data =  data[3:]

print(np.shape(train_data))
print(train_data.dtype)
#test_data = MovingMNIST[9000:10000]
norm=Normalize(np.shape(train_data)[1])

normalized_train=norm.normalize(train_data)[0]
normalized_val=norm.normalize_other(val_data)

print(np.shape(normalized_train))
print(np.shape(normalized_val))

#lest visualize the train data and test data
#seq,temp(feature),frame,depth
save_comparison_plot(normalized_train[0][0][0],normalized_train[0][0][1],normalized_train[0][0][2],file="seq1_train",depth=0)
save_comparison_plot(normalized_train[1][0][0],normalized_train[1][0][1],normalized_train[1][0][2],file="seq2_train",depth=0)
save_comparison_plot(normalized_train[2][0][0],normalized_train[2][0][1],normalized_train[2][0][2],file="seq3_train",depth=0)
save_comparison_plot(normalized_train[0][0][0],normalized_train[0][0][1],normalized_train[0][0][2],file="seq1_test",depth=0)

#Dimensions are (Batch size, channels, frames, depth, length, width)
#shape is (B, 19, F, 47, 265, 442)  channels are nothing but features like temp o2 etc


def collate(batch):
    batch = torch.tensor(batch)
    batch = batch.to(device)
    return batch[:,:,0:2], batch[:,:,2]


# Training Data Loader
train_loader = DataLoader(normalized_train, shuffle=False,
                        batch_size=3, collate_fn=collate)

# Validation Data Loader
val_loader = DataLoader(normalized_val, shuffle=False,
                        batch_size=1, collate_fn=collate)

# The input video frames are grayscale, thus single channel
model = Seq2Seq(num_channels=np.shape(normalized_train)[1], num_kernels=64,
kernel_size=(3, 3, 3), padding=(1, 1, 1), activation="relu",
frame_size=(10, 32, 32), num_layers=2).to(device)


print(model.conv.parameters())
optim = Adam(model.parameters(), lr=1e-4)
print(optim.__module__)

criterion = nn.MSELoss()

num_epochs = 4

for epoch in range(1, num_epochs + 1):
    train_loss = 0
    model.train()
    for batch_num, (input, target) in enumerate(train_loader, 1):
        print(f"Batch {batch_num}:")

        # Print input and target details
        print(f"Input Shape: {input.shape}")
        print(f"Target Shape: {target.shape}")
        print(f"Input (Min: {input.min().item()}, Max: {input.max().item()})")
        print(f"Target (Min: {target.min().item()}, Max: {target.max().item()})")
        with torch.autograd.detect_anomaly():
            output = model(input)

            print(f"Output Shape: {output.shape}")
            print(f"Output (Min: {output.min().item()}, Max: {output.max().item()})")

            if torch.isnan(output).any():
                print("NaN detected in output")
                print("Input stats:", torch.min(input).item(), torch.max(input).item())
                print("Latest conv layer weights:", torch.min(model.conv.weight).item(),
                      torch.max(model.conv.weight).item())

            loss = criterion(output.flatten(), target.flatten())

            print(f"Loss: {loss.item()}")

            loss.backward()
            optim.step()
            optim.zero_grad()
        train_loss += loss.item()
    train_loss /= len(train_loader.dataset)
    print()
    val_loss = 0
    model.eval()
    #     with torch.no_grad():
    #         for input, target in val_loader:
    #             output = model(input)
    #             loss = criterion(output.flatten(), target.flatten())
    #             val_loss += loss.item()
    #     val_loss /= len(val_loader.dataset)

    print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f}\n".format(
        epoch, train_loss, val_loss))



predictions = []
test_loader = val_loader
    # Iterate through the test dataset
with torch.no_grad():
    for input_frames, target_frame in test_loader:  # Replace 'test_loader' with your actual test loader
        print("Processing a batch...")
        input_frames = input_frames.to(device)  # Move input frames to device
        target_frame = target_frame.to(device)  # Move target frame to device

        # Predict the next frame
        predicted_frame = model(input_frames)

        # Store predictions and optionally the target for comparison
        predictions.append((predicted_frame.cpu(), target_frame.cpu()))
        print(f"Number of predictions so far: {len(predictions)}")

        save_comparison_plot(input_frames[0][0][0], input_frames[0][0][1], target_frame[0][0],predicted_frame[0][0],
                             file="predictedframeatend", depth=0)
        # calculate_metrics(target_frame[0][0],predicted_frame[0][0])
print("Comparison completed and plots saved.")

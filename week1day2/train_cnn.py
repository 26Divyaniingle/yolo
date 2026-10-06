import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

x = torch.tensor([1, 2, 3])

print(x)
print(x.shape)

#load cifar dataset
transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True, #give images to train the model
    download=True,
    transform=transform
)


validation_dataset = datasets.CIFAR10(
    root="./data",
    train=False, #give image to check the performance of the model
    download=True,
    transform=transform
)

print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))

#added dataloader to load the data in batches
batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True #shuffles the data in each epoch
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False
)


##create cnn class
class SmallCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.fc = nn.Linear(
            64 * 8 * 8,
            10
        )

    #forward function
    def forward(self, x):

        x = self.conv1(x)

        x = torch.relu(x)

        x = self.pool(x)

        x = self.conv2(x)

        x = torch.relu(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

model = SmallCNN()

print(model )

#loss function
criterion = nn.CrossEntropyLoss()

#optimizer to update the weights
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)
model = model.to(device)

#training loop
num_epochs = 3

train_losses = []
validation_losses = []


for epoch in range(num_epochs):
    

    model.train()

    running_train_loss = 0.0


    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)


        # Clear previous gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images)


        # Calculate loss
        loss = criterion(outputs, labels)


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        running_train_loss += loss.item()


    average_train_loss = (
        running_train_loss / len(train_loader)
    )


    train_losses.append(average_train_loss)

    #validation
    model.eval()

    running_validation_loss = 0.0


    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_validation_loss += loss.item()


    average_validation_loss = (
        running_validation_loss / len(validation_loader)
    )

    validation_losses.append(average_validation_loss)


    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Training Loss: {average_train_loss:.4f} "
        f"Validation Loss: {average_validation_loss:.4f}"
    )

#plot loss 
plt.plot(
    range(1, num_epochs + 1),
    train_losses,
    label="Train Loss"
)


plt.plot(
    range(1, num_epochs + 1),
    validation_losses,
    label="Validation Loss"
)


plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

plt.show()
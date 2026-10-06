import os

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


# ==========================================
# STEP 1: CREATE 3-CLASS IMAGE FOLDER DATASET
# ==========================================

DATASET_DIR = "./data/dataset_3class"

CLASSES = ["airplane", "automobile", "bird"]

TARGET_LABELS = [0, 1, 2]

TRAIN_IMAGES_PER_CLASS = 150
VAL_IMAGES_PER_CLASS = 50


def prepare_3class_dataset():

    if os.path.exists(DATASET_DIR):

        print(
            f"Dataset already exists at: {DATASET_DIR}"
        )

        return

    print("Creating 3-class Image Folder from CIFAR-10...")

    # Download CIFAR-10 training set
    raw_train = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True
    )

    # Download CIFAR-10 test set
    raw_val = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True
    )

    # Create folders
    for split in ["train", "val"]:

        for cls_name in CLASSES:

            os.makedirs(
                os.path.join(
                    DATASET_DIR,
                    split,
                    cls_name
                ),
                exist_ok=True
            )

    # ------------------------------------------
    # Save training images
    # ------------------------------------------

    counts = {
        label: 0
        for label in TARGET_LABELS
    }

    for img, label in raw_train:

        if (
            label in TARGET_LABELS
            and counts[label] < TRAIN_IMAGES_PER_CLASS
        ):

            cls_name = CLASSES[label]

            image_path = os.path.join(
                DATASET_DIR,
                "train",
                cls_name,
                f"{counts[label]}.jpg"
            )

            img.save(image_path)

            counts[label] += 1

        if all(
            count >= TRAIN_IMAGES_PER_CLASS
            for count in counts.values()
        ):
            break

    # ------------------------------------------
    # Save validation images
    # ------------------------------------------

    val_counts = {
        label: 0
        for label in TARGET_LABELS
    }

    for img, label in raw_val:

        if (
            label in TARGET_LABELS
            and val_counts[label] < VAL_IMAGES_PER_CLASS
        ):

            cls_name = CLASSES[label]

            image_path = os.path.join(
                DATASET_DIR,
                "val",
                cls_name,
                f"{val_counts[label]}.jpg"
            )

            img.save(image_path)

            val_counts[label] += 1

        if all(
            count >= VAL_IMAGES_PER_CLASS
            for count in val_counts.values()
        ):
            break

    print("\nDataset created successfully!")

    print(
        "Train: 150 images per class "
        "(Total: 450)"
    )

    print(
        "Val:   50 images per class "
        "(Total: 150)"
    )


# ==========================================
# STEP 2: DATA TRANSFORMS & LOADERS
# ==========================================

def get_dataloaders(
    use_augmentation=False,
    batch_size=32
):

    # ImageNet normalization
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # ------------------------------------------
    # Training transformation
    # ------------------------------------------

    if use_augmentation:

        train_transform = transforms.Compose([

            transforms.Resize(
                (224, 224)
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomRotation(
                degrees=15
            ),

            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2
            ),

            transforms.ToTensor(),

            norm
        ])

    else:

        train_transform = transforms.Compose([

            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor(),

            norm
        ])

    # ------------------------------------------
    # Validation transformation
    # ------------------------------------------

    val_transform = transforms.Compose([

        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        norm
    ])

    # ------------------------------------------
    # ImageFolder datasets
    # ------------------------------------------

    train_data = datasets.ImageFolder(

        os.path.join(
            DATASET_DIR,
            "train"
        ),

        transform=train_transform
    )

    val_data = datasets.ImageFolder(

        os.path.join(
            DATASET_DIR,
            "val"
        ),

        transform=val_transform
    )

    # ------------------------------------------
    # DataLoaders
    # ------------------------------------------

    train_loader = DataLoader(

        train_data,

        batch_size=batch_size,

        shuffle=True,

        num_workers=0
    )

    val_loader = DataLoader(

        val_data,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0
    )

    return (
        train_loader,
        val_loader,
        train_data.classes
    )


# ==========================================
# STEP 3: MODEL SETUP & TRANSFER LEARNING
# ==========================================

def build_model(num_classes=3):

    # ------------------------------------------
    # 1. Load pretrained ResNet18
    # ------------------------------------------

    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(
        weights=weights
    )

    # ------------------------------------------
    # 2. Freeze backbone
    # ------------------------------------------

    for param in model.parameters():

        param.requires_grad = False

    # ------------------------------------------
    # 3. Replace classifier
    # ------------------------------------------

    num_ftrs = model.fc.in_features

    print(
        "Original classifier input features:",
        num_ftrs
    )

    model.fc = nn.Linear(
        num_ftrs,
        num_classes
    )

    return model


# ==========================================
# STEP 4: TRAINING & EVALUATION
# ==========================================

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=5,
    lr=0.001
):

    criterion = nn.CrossEntropyLoss()

    # Only train classifier
    optimizer = optim.Adam(
        model.fc.parameters(),
        lr=lr
    )

    history = {

        "train_loss": [],
        "train_acc": [],

        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(num_epochs):

        # ==================================
        # TRAIN PHASE
        # ==================================

        model.train()

        train_loss = 0.0

        train_correct = 0

        total_train = 0

        for images, labels in train_loader:

            images = images.to(device)

            labels = labels.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(
                outputs,
                labels
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate loss
            train_loss += (
                loss.item()
                * images.size(0)
            )

            # Predictions
            preds = outputs.argmax(1)

            train_correct += (
                (preds == labels)
                .sum()
                .item()
            )

            total_train += labels.size(0)

        epoch_train_loss = (
            train_loss / total_train
        )

        epoch_train_acc = (
            train_correct
            / total_train
            * 100.0
        )

        # ==================================
        # VALIDATION PHASE
        # ==================================

        model.eval()

        val_loss = 0.0

        val_correct = 0

        total_val = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)

                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss += (
                    loss.item()
                    * images.size(0)
                )

                preds = outputs.argmax(1)

                val_correct += (
                    (preds == labels)
                    .sum()
                    .item()
                )

                total_val += labels.size(0)

        epoch_val_loss = (
            val_loss / total_val
        )

        epoch_val_acc = (
            val_correct
            / total_val
            * 100.0
        )

        # ==================================
        # SAVE HISTORY
        # ==================================

        history["train_loss"].append(
            epoch_train_loss
        )

        history["train_acc"].append(
            epoch_train_acc
        )

        history["val_loss"].append(
            epoch_val_loss
        )

        history["val_acc"].append(
            epoch_val_acc
        )

        # ==================================
        # PRINT RESULT
        # ==================================

        print(

            f"Epoch [{epoch + 1}/{num_epochs}] | "

            f"Train Loss: "
            f"{epoch_train_loss:.4f}, "

            f"Train Acc: "
            f"{epoch_train_acc:.1f}% | "

            f"Val Loss: "
            f"{epoch_val_loss:.4f}, "

            f"Val Acc: "
            f"{epoch_val_acc:.1f}%"
        )

    return history


# ==========================================
# STEP 5: MAIN
# ==========================================

def main():

    # --------------------------------------
    # Create dataset
    # --------------------------------------

    prepare_3class_dataset()

    # --------------------------------------
    # Device
    # --------------------------------------

    device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nUsing device: {device}"
    )

    epochs = 5

    # ======================================
    # EXPERIMENT 1
    # WITHOUT AUGMENTATION
    # ======================================

    print("\n" + "=" * 60)

    print(
        "EXPERIMENT 1: "
        "WITHOUT AUGMENTATION"
    )

    print("=" * 60)

    (
        train_loader_no_aug,
        val_loader,
        class_names
    ) = get_dataloaders(
        use_augmentation=False
    )

    print(
        f"Classes: {class_names}"
    )

    model_no_aug = build_model(
        num_classes=len(class_names)
    )

    model_no_aug = model_no_aug.to(device)

    hist_no_aug = train_model(

        model_no_aug,

        train_loader_no_aug,

        val_loader,

        device,

        num_epochs=epochs
    )

    # ======================================
    # EXPERIMENT 2
    # WITH AUGMENTATION
    # ======================================

    print("\n" + "=" * 60)

    print(
        "EXPERIMENT 2: "
        "WITH AUGMENTATION"
    )

    print(
        "Flip + Rotation + ColorJitter"
    )

    print("=" * 60)

    (
        train_loader_aug,
        _,
        _
    ) = get_dataloaders(
        use_augmentation=True
    )

    # IMPORTANT:
    # Create a completely fresh model

    model_aug = build_model(
        num_classes=len(class_names)
    )

    model_aug = model_aug.to(device)

    hist_aug = train_model(

        model_aug,

        train_loader_aug,

        val_loader,

        device,

        num_epochs=epochs
    )

    # ======================================
    # FINAL REPORT
    # ======================================

    print("\n" + "=" * 60)

    print(
        "FINAL ACCURACY REPORT"
    )

    print("=" * 60)

    print(
        f"{'Condition':<25}"
        f"| {'Train Acc':<15}"
        f"| {'Val Acc':<15}"
    )

    print("-" * 60)

    print(

        f"{'Without Augmentation':<25}"
        f"| "
        f"{hist_no_aug['train_acc'][-1]:.2f}%"
        f"{'':<8}"
        f"| "
        f"{hist_no_aug['val_acc'][-1]:.2f}%"
    )

    print(

        f"{'With Augmentation':<25}"
        f"| "
        f"{hist_aug['train_acc'][-1]:.2f}%"
        f"{'':<8}"
        f"| "
        f"{hist_aug['val_acc'][-1]:.2f}%"
    )

    print("-" * 60)

    # ======================================
    # PLOT
    # ======================================

    plt.figure(
        figsize=(12, 5)
    )

    # --------------------------------------
    # LOSS
    # --------------------------------------

    plt.subplot(1, 2, 1)

    plt.plot(
        range(1, epochs + 1),
        hist_no_aug["train_loss"],
        "r--",
        label="Train Loss - No Aug"
    )

    plt.plot(
        range(1, epochs + 1),
        hist_no_aug["val_loss"],
        "r-",
        label="Val Loss - No Aug"
    )

    plt.plot(
        range(1, epochs + 1),
        hist_aug["train_loss"],
        "b--",
        label="Train Loss - Aug"
    )

    plt.plot(
        range(1, epochs + 1),
        hist_aug["val_loss"],
        "b-",
        label="Val Loss - Aug"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        "Loss Comparison"
    )

    plt.legend()

    plt.grid(True)

    # --------------------------------------
    # ACCURACY
    # --------------------------------------

    plt.subplot(1, 2, 2)

    plt.plot(
        range(1, epochs + 1),
        hist_no_aug["val_acc"],
        "r-o",
        label="Val Acc - No Aug"
    )

    plt.plot(
        range(1, epochs + 1),
        hist_aug["val_acc"],
        "b-o",
        label="Val Acc - Aug"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy (%)")

    plt.title(
        "Validation Accuracy Comparison"
    )

    plt.legend()

    plt.grid(True)

    # --------------------------------------
    # SAVE GRAPH
    # --------------------------------------

    os.makedirs(
        "./output",
        exist_ok=True
    )

    plot_path = (
        "./output/"
        "transfer_learning_comparison.png"
    )

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"\nComparison plot saved to:"
        f"\n{plot_path}"
    )

    plt.show()

    print(
        "\n[Done] "
        "All experiments completed successfully!"
    )


# ==========================================
# PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":

    main()
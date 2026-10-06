import torch
from torchvision import models
from PIL import Image


weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)
model.eval()

print("ResNet18 loaded successfully!")

print("\n========== RESNET18 ARCHITECTURE ==========\n")

print(model)
preprocess = weights.transforms()

preprocess = weights.transforms()

categories = weights.meta["categories"]

print("\nNumber of classes:", len(categories))

for i in range(1, 6):

    image_path = f"images/image{i}.jpg"

    print("\nProcessing:", image_path)

    image = Image.open(image_path).convert("RGB")

    input_tensor = preprocess(image)

    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    predicted_class = output.argmax(1).item()

    prediction = categories[predicted_class]

    print("Prediction:", prediction)

print("Image loaded successfully!")
print("Image size:", image.size)

input_tensor = preprocess(image)
print("Tensor shape:", input_tensor.shape)

#add batch dimension
input_batch = input_tensor.unsqueeze(0)
print("Batch shape:", input_batch.shape)

with torch.no_grad():
    output = model(input_batch)

print("Output shape:", output.shape)

#get the predicted class
predicted_class = output.argmax(1).item()

prediction = categories[predicted_class]
print("Prediction:", prediction)

import torch
import cv2
import numpy as np
import matplotlib
import ultralytics

print("===== ENVIRONMENT CHECK =====")

print("Python packages:")
print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)
print("Matplotlib:", matplotlib.__version__)
print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)

print("\n===== GPU CHECK =====")

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU not available")
    print("Using CPU")
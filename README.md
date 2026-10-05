# Computer Vision & YOLO Projects

This repository contains computer vision workflows, OpenCV fundamentals, and video processing pipelines.

## Project Structure

- `week1day1/`: OpenCV exercises covering image reading, manipulation, transformations, and drawing:
  - `imagebasics.py`
  - `opencv2.py`
  - `opencv_operations.py`
- `video_processing.py`: Real-time video processing with frame conversion, FPS calculation, and output saving.
- `setup_test.py`: Environment check script verifying PyTorch, CUDA/GPU support, OpenCV, Ultralytics, and NumPy.
- `images/`: Input test images.
- `output/`: Processed image and video outputs.
- `videos/`: Input video files.

## Getting Started

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the environment verification test:
   ```bash
   python setup_test.py
   ```

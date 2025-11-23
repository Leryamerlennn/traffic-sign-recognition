# Detection part of trafic-sign-recognition 


### About the Project

This project is a computer vision system for real-time object detection and analysis using a camera. The system includes a complete image processing pipeline: preprocessing, color segmentation, and contour analysis.


### Project Structure

``` text
development/detection/
├── preprocess.py          # Image preprocessing
├── color_segmentation.py  # Color segmentation
├── contour_check.py       # Contour analysis and filtering
└── video_catch.py         # Main camera module, created to check

```



### Functionality

1. Image Preprocessing (preprocess.py)

Resizing with aspect ratio preservation and padding
Contrast enhancement using CLAHE
Gaussian blur for noise reduction
Illumination correction based on Otsu algorithm

2. Color Segmentation (color_segmentation.py)

Object detection by colors: 
Red (two HSV ranges)
Blue
White (by brightness and saturation)
Morphological operations for mask cleaning
Contour detection and filtering

3. Contour Analysis (contour_check.py)

Geometric characteristics analysis:
Area and perimeter
Roundness and convexity
Aspect ratio
Vertex count
Filtering by:

Size and position
Intersection with other contours
Corner placement

4. Camera Processing (video_catch.py)

Webcam video capture
Real-time processing
Results visualization

### Usage

```bash
cd development/detection
python video_catch.py

```

### Contours: 

Press ESC to exit
Detected objects are highlighted with green contours

Requirements

```bash
pip install opencv-python numpy matplotlib
```

### Parameter Configuration

Color Ranges (in HSV):

Red: [0-10] and [170-180]
Blue: [100-140]
White: Brightness > 200, Saturation < 50
Contour Filtering:

Minimum area: 600 pixels
Maximum area: 60000 pixels
Maximum aspect ratio: 4.0

### Key Features

Real-time processing with webcam integration
Robust preprocessing for various lighting conditions
Multi-color detection capability
Advanced contour analysis with geometric filtering
Modular architecture for easy customization
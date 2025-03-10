# Convolutional Neural Network (CNN) Image Processing

## Overview
This project demonstrates the use of a Convolutional Neural Network (CNN) for image processing. It applies three main operations:
1. **Convolution**: Uses an edge-detection filter to enhance features.
2. **ReLU Activation**: Applies non-linearity to remove negative values.
3. **Max Pooling**: Reduces the image size while retaining important features.

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

## How to Run
1. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib
   ```
2. Place your image (`image.jpg`) in the same directory as the script.
3. Run the Python script:
   ```bash
   python cnn_image_processing.py
   ```
   Or run the Jupyter Notebook if using `.ipynb`.

## Expected Output
The script will generate and display:
1. The **original grayscale image**.
2. The **convolved image** after edge detection.
3. The **activated image** after ReLU function.
4. The **pooled image** after max pooling.

## Notes
- You can replace `image.jpg` with any other image, but ensure it's in the same directory.
- The kernel used is a basic edge-detection filter, but you can modify it for different effects.

## License
This project is open-source and can be modified for educational and research purposes.


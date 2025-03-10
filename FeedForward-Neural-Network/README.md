# Feedforward Neural Network for MNIST Classification

This project demonstrates how to build, train, and evaluate a simple **Feedforward Neural Network** (FNN) using **TensorFlow** and **Keras** to classify handwritten digits from the **MNIST dataset**.

## 📌 Project Overview

- Loads the **MNIST dataset** (28x28 grayscale images of digits 0-9).
- Normalizes the image pixel values to the range **[0,1]**.
- Builds a **Neural Network** with:
  - **Flatten layer**: Converts 2D images into a 1D array.
  - **Dense layer** (128 neurons, ReLU activation): Learns important patterns.
  - **Dense output layer** (10 neurons, Softmax activation): Outputs probabilities for each digit.
- Compiles the model using:
  - **Adam Optimizer**
  - **Sparse Categorical Crossentropy Loss**
  - **Sparse Categorical Accuracy Metric**
- Trains the model for **5 epochs**.
- Evaluates the model on the test dataset and prints the accuracy.

## 📂 Files

- `Feedforward_NN_MNIST.ipynb` - Jupyter Notebook with full implementation.
- `README.md` - This file, explaining the project.

## 🚀 Requirements

Ensure you have **Python 3.7+** and install the required libraries using:

```sh
pip install tensorflow numpy matplotlib
```

## ▶️ How to Run

1. Open `Feedforward_NN_MNIST.ipynb` in **Jupyter Notebook** or **Google Colab**.
2. Run the cells step-by-step to train and evaluate the model.

## 📊 Expected Output

After training for **5 epochs**, the model should achieve **~97-98% accuracy** on the MNIST dataset.

## 🤖 Future Improvements

- Increase the number of neurons or layers for better accuracy.
- Try different activation functions and optimizers.
- Train for more epochs or use early stopping.

---

Made using TensorFlow & Keras 🚀

---

### 👨‍💻 Author
[Shivani Tyagi](https://github.com/shivitg) - Data Science Enthusiast 🚀
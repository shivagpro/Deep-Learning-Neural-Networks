# ANN-Churn-Prediction

This repository demonstrates how to implement an Artificial Neural Network (ANN) for customer churn prediction in the banking sector. Using deep learning techniques, the model predicts whether a customer will exit the bank based on key features like account balance, credit score, and tenure.

## 📂 Project Structure

```
ANN-Churn-Prediction/
│-- app.py                                  # Flask app for deployment (if applicable)
│-- Churn_Modelling.csv                     # Dataset for training and testing
│-- Customer Churn Training using ANN.ipynb # Jupyter Notebook for training
│-- label_encoder_gender.pkl                 # Saved LabelEncoder for gender
│-- model.h5                                 # Trained ANN model
│-- onehot_encoder_geo.pkl                   # Saved OneHotEncoder for geography
│-- Prediction using ANN.ipynb               # Jupyter Notebook for predictions
│-- README.md                                # Project documentation
│-- requirements.txt                         # Python dependencies
│-- scaler.pkl                               # Saved StandardScaler for preprocessing
```

## 🚀 Project Overview

Customer churn prediction is crucial for banks to retain valuable clients. This project employs an ANN model built using TensorFlow/Keras to classify customers as either staying or leaving.

### **Key Features**
- **Data Preprocessing:** Encoding categorical variables, feature scaling.
- **ANN Model Training:** Implemented using TensorFlow/Keras.
- **Model Evaluation:** Using accuracy metrics.
- **Predictions:** Making predictions on new customer data.
- **Serialization:** Saving trained encoders and scalers for future use.

## 📊 Dataset
The dataset (`Churn_Modelling.csv`) contains 10,000 customer records with the following features:
- `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`
- `Exited` (Target Variable: 1 = Churn, 0 = Retained)

## 🔧 Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SHIVITG/Deep-Learning-Neural-Networks.git
   cd ANN-Churn-Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Jupyter Notebook** for training and prediction
   ```bash
   jupyter notebook
   ```

## 🧠 Model Training
The ANN model is built using the following architecture:
- **Input Layer**: 11 features
- **Hidden Layers**:
  - Dense layer (64 neurons, ReLU activation)
  - Dense layer (32 neurons, ReLU activation)
- **Output Layer**: 1 neuron (Sigmoid activation for binary classification)

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

The model is compiled and trained with:

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)
```

## 🔍 Making Predictions
Use the `Prediction using ANN.ipynb` notebook to test the trained model on new customer data. Example:
```python
model = load_model('model.h5')
input_data = np.array([[600, 1, 40, 3, 60000, 2, 1, 1, 50000]])  # Preprocessed Input
prediction = model.predict(input_data)
print('Churn' if prediction > 0.5 else 'No Churn')
```

## 📌 Future Improvements
- Hyperparameter tuning for better performance
- Deploying the model via Flask (`app.py`)
- Implementing advanced feature engineering

---
### 👨‍💻 Author
[Shivani Tyagi](https://github.com/shivitg) - Data Science Enthusiast 🚀


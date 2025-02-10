# 🍎 Fruit Classification Using CNN (Fruits-360 Dataset)

## 📌 Project Overview
This project trains a **Convolutional Neural Network (CNN)** to classify different types of fruits using the **Fruits-360 dataset** from Kaggle. The model leverages **TensorFlow, Keras**, and **transfer learning** with **VGG16** for better accuracy. It can predict fruit categories from images, making it useful for **fruit recognition, quality control, and educational tools**.

---

## 📂 Dataset Details
- **Dataset**: [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits)
- **Image Size**: 100x100 pixels
- **Classes**: Multiple fruit categories
- **Structure**:
  - `Training/` – Images for training the model
  - `Test/` – Images for validation/testing

---

## 🏗 Model Architecture
- **Pretrained Model**: VGG16 (Feature extraction)
- **Custom Layers**:
  - `Flatten()`
  - `Dense(256, activation='relu')`
  - `Dropout(0.5)`
  - `Dense(num_classes, activation='softmax')`
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam (LR = 0.0001)
- **Epochs**: 10
- **Batch Size**: 32

---

## 📊 Results & Performance
- **Training & Validation Accuracy**: 📈 Tracked using `matplotlib`
- **Loss Reduction**: 📉 Achieved using `EarlyStopping`
- **Testing**: Predicts fruit type from custom images

---

## 🚀 How to Use
### 1️⃣ Clone Repository
```bash
git clone https://github.com/soy-praveen/fruit-classifier.git
pip install tensorflow keras numpy matplotlib
python train.py
python predict.py --image path_to_fruit.jpg
```
## For predicting a new image
```bash
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("fruit_classifier.h5")

# Load and preprocess image
img_path = "path_to_fruit_image.jpg"
img = image.load_img(img_path, target_size=(100, 100))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict fruit class
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
class_labels = list(train_generator.class_indices.keys())
predicted_label = class_labels[predicted_class]

print(f"Predicted Fruit: {predicted_label}")
```

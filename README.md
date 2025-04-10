# Digit-Recognizer-Numpy
# Handwritten Digit Recognizer using NumPy (from Scratch)

This is a simple yet powerful handwritten digit classifier trained on the [Kaggle Digit Recognizer dataset](https://www.kaggle.com/competitions/digit-recognizer) using a fully-connected neural network implemented **from scratch** in **NumPy**.

It achieves **92% accuracy** on the training set using only:
- One hidden layer with 64 neurons
- ReLU + Softmax activation functions
- Cross-entropy loss
- Mini-batch gradient descent

---

## 🔥 Why this project matters
Most tutorials use TensorFlow or PyTorch. But building everything from scratch using NumPy gives you a **deep understanding** of how neural networks work under the hood.

If you're a student, beginner, or someone who wants to master ML fundamentals — this is for you.

---

## 📁 Dataset
We use the **Digit Recognizer** dataset from Kaggle:
- **Train:** 42,000 labeled digit images (28x28 grayscale)
- **Test:** 28,000 unlabeled digit images

### 🧠 Target:
Train a neural network to classify digits (0–9).

---

## 🧠 Model Architecture
| Layer         | Details                    |
|---------------|-----------------------------|
| Input         | 784 neurons (28x28 pixels) |
| Hidden Layer  | 64 neurons, ReLU activation|
| Output Layer  | 10 neurons, Softmax        |

---

## 🚀 How to Run (Locally)

1. Clone this repo:
```bash
git clone https://github.com/Nouhid665/digit-recognizer-numpy.git
cd digit-recognizer-numpy
```

2. Install dependencies:
```bash
pip install numpy matplotlib pandas
```

3. Download dataset:
- Go to [Digit Recognizer - Kaggle](https://www.kaggle.com/competitions/digit-recognizer/data)
- Download and place `train.csv` and `test.csv` in your project folder

4. Run the notebook or script:
```bash
python train_model.py  # or open notebook.ipynb
```

---

## 📊 Results
- ✅ Accuracy: **92% on training set**
- 🧪 Generalizes well with very little overfitting
- 🔍 Each digit can be visualized and tested using the `test_prediction()` function

---

## 📸 Sample Predictions
```python
test_prediction(index=13, W1, B1, W2, B2)
```
Shows the image and prints predicted vs actual label.

---

## 📌 Key Learnings
| Mistake | Fix |
|--------|-----|
| No mini-batch training | Added batch size = 64 |
| No ReLU derivative | Implemented `relu_derivative()` |
| Unstable softmax | Used `np.clip()` in `entropy_loss` to avoid log(0) |
| Bad weight init | Used He initialization (`np.sqrt(2./in_size)`) |

---

## 🧠 Author
- **Nouhid Siddiqui**  
- GitHub: [Nouhid665](https://github.com/Nouhid665)

---

## 📌 To Do
- [ ] Add evaluation on test set
- [ ] Export predictions to CSV for Kaggle submission
- [ ] Try adding another hidden layer

---

## 📄 License
This project is open-source and free to use. No frameworks. Pure learning.


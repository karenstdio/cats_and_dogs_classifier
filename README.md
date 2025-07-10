#Cats vs Dogs Classifier using CNN (TensorFlow)

This project uses a Convolutional Neural Network (CNN) to classify images of cats and dogs using the TensorFlow/Keras library.

## Dataset

The dataset is the official `cats_and_dogs_filtered` set provided by Google:
- Train and validation directories
- Two classes: `cats` and `dogs`

## How to Run

### 1. Clone this Repository

```bash
git clone https://github.com/your_username/cats-vs-dogs-classifier.git
cd cats-vs-dogs-classifier
```

### 2. Install Dependencies

```bash
pip install tensorflow matplotlib numpy pillow
```

### 3. Run the Script

```bash
python cats_and_dogs_classifier.py
```

## Model Architecture

- 3 convolutional layers with max pooling
- Flatten + Dense layer with ReLU
- Final sigmoid activation for binary classification

## Output

- Model training for 20 epochs
- Accuracy evaluation on validation set
- Random test image classification and visualization

## Example Output

```
Validation Accuracy: 87.50%
Predicted: Dog (0.95)
```

## Notes

- You can adjust `img_size` and `epochs` to experiment.
- Ensure `cats_and_dogs_filtered.zip` is successfully downloaded and extracted.

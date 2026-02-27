import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

from src.rational_classifier import RationalClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import os
os.makedirs("results", exist_ok=True)

np.random.seed(42)

########################
##### LOADING DATA #####
########################

print("Loading MNIST dataset...")

mnist = fetch_openml('mnist_784', version=1)
X = mnist.data
y = mnist.target.astype(int)

# Randomly select subset
subset_size = 2000
indices = np.random.choice(len(X), subset_size, replace=False)
X_subset = X.iloc[indices]
y_subset = y.iloc[indices]

# Normalize features
scaler = MinMaxScaler()
X_subset_scaled = scaler.fit_transform(X_subset)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_subset_scaled, y_subset, test_size=0.2, random_state=42
)

print("Data preparation completed.")


########################
###### TRAINING ########
########################

print("Applying PCA...")

n_components = 22
pca = PCA(n_components=n_components)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("Initializing Rational Classifier...")

numerator_degree = 2
denominator_degree = 1

classifier = RationalClassifier(
    numerator_degree,
    denominator_degree,
    n_components
)

print("Training model...")
classifier.fit(X_train_pca, y_train)


########################
######## TESTING #######
########################

print("Predicting on test set...")
y_pred = classifier.predict(X_test_pca)


########################
###### EVALUATION ######
########################

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

#ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
#plt.title("Confusion Matrix")
#plt.show()
# Make sure results folder exists
os.makedirs("results", exist_ok=True)

# Confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")  # Save figure
plt.show()

report = classification_report(y_test, y_pred, digits=4)
print(report)

with open("results/classification_report.txt", "w") as f:
    f.write(report)


########################
##### VISUALIZATION ####
########################

def visualize_images(num_images=20):
    plt.figure(figsize=(15, 10))

    for i in range(num_images):
        new_image = X_test[i]
        true_digit = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
        predicted_digit = y_pred[i]

        reshaped_image = new_image.reshape(28, 28)

        plt.subplot(4, 5, i + 1)
        plt.imshow(reshaped_image, cmap='gray')
        plt.title(f"True: {true_digit}, Pred: {predicted_digit}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("results/sample_predictions.png")  # Save figure
    plt.show()


visualize_images(20)
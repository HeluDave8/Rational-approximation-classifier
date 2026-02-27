import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from scipy.optimize import linprog
from .utils import generate_rational_function_matrix, generate_multi_indices, r_multi_indices
from .optimization import solve_lp, bisection_method
class RationalClassifier:
    def __init__(self, numerator_degree, denominator_degree, n_components, delta=1e-5, precision=1e-3):
        self.numerator_degree = numerator_degree
        self.denominator_degree = denominator_degree
        self.n_components = n_components
        self.delta = delta
        self.precision = precision
        self.alpha = None
        self.beta = None
        self.z = None
        self.classifiers = {}

    def fit(self, X, y):
        """
        Train the rational classifier by solving for optimal alpha, beta, and z.
        """
        for cls in range(10):
            # Binarize the labels: current digit vs. other digits (One-vs-All)
            y_binary = np.where(y == cls, 1, 0)

            # Generate rational function matrices
            G_matrix, H_matrix = generate_rational_function_matrix(
                self.numerator_degree, self.denominator_degree, self.n_components, X
            )

            # Use bisection method to find optimal z
            self.z = bisection_method(
                uL=0, uH=100, G_matrix=G_matrix, H_matrix=H_matrix, y=y_binary, precision=self.precision
            )

            # Solve for alpha and beta at the optimal z
            result = solve_lp(self.z, G_matrix, H_matrix, y_binary, delta=self.delta)
            if not result.success:
                raise ValueError(f"Linear programming failed to converge for digit {cls}.")

            # Store the coefficients for each digit's classifier
            self.classifiers[cls] = {
                'alpha': result.x[1:1 + G_matrix.shape[1]],
                'beta': result.x[1 + G_matrix.shape[1]:]
            }

    def predict(self, X):
        """
        Predict class labels for input data X.
        """
        predictions = []
        for x in X:
            class_scores = {}

            # Compute scores for each classifier (digit 0-9)
            for cls, coeffs in self.classifiers.items():
                alpha = coeffs['alpha']
                beta = coeffs['beta']

                # Generate rational function matrices for prediction
                G_matrix, H_matrix = generate_rational_function_matrix(
                    self.numerator_degree, self.denominator_degree, self.n_components, [x]
                )

                # Compute rational function values
                numerator = np.dot(G_matrix, alpha)
                denominator = np.dot(H_matrix, beta)+1e-5 
                rational_value = numerator / denominator

                class_scores[cls] = rational_value

            # Choose the class (digit) with the highest score
            predicted_digit = max(class_scores, key=class_scores.get)
            predictions.append(predicted_digit)

        return np.array(predictions)

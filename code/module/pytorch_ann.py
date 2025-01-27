import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class PyTorchANN(BaseEstimator, ClassifierMixin):
    """
    A simple PyTorch-based ANN classifier in a scikit-learn style wrapper.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        num_hidden_layers=1,
        epochs=20,
        batch_size=32,
        learning_rate=0.001,
        device=None
    ):
        """
        :param input_dim:       Number of input features.
        :param hidden_dim:      Number of neurons in each hidden layer.
        :param num_hidden_layers: How many hidden layers to use.
        :param epochs:          Number of training epochs.
        :param batch_size:      Batch size for training.
        :param learning_rate:   Learning rate for the optimizer.
        :param device:          Device ('cuda' or 'cpu'). If None, auto-detect.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if device is not None else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # The model will be created in the 'fit' method after we know #classes
        self.model = None

    def _build_network(self, num_classes=2):
        """
        Dynamically build a feed-forward network based on constructor parameters.
        For binary classification with a single output, we’ll use a single neuron 
        and a Sigmoid at the output. For multi-class, we’ll use num_classes output neurons.
        """

        # If we're doing binary classification, we can use 1 output neuron + Sigmoid
        if num_classes == 2:
            output_dim = 1
        else:
            output_dim = num_classes

        # Build a list of layers
        layers = []
        # Input -> first hidden
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        layers.append(nn.ReLU())

        # Additional hidden layers if requested
        for _ in range(self.num_hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())

        # Last hidden -> output
        layers.append(nn.Linear(self.hidden_dim, output_dim))

        return nn.Sequential(*layers)

    def fit(self, X, y):
        """
        Fit the PyTorch model on the given training data.
        """
        # Convert X, y to torch Tensors
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32)

        # If your target is multi-class, you may need to handle it differently
        # For binary classification, we’ll treat y as float [0 or 1].
        num_classes = len(torch.unique(y_tensor))
        self.model = self._build_network(num_classes=num_classes).to(self.device)

        # Define loss & optimizer
        if num_classes == 2:
            criterion = nn.BCEWithLogitsLoss()  # For single-output Sigmoid
        else:
            criterion = nn.CrossEntropyLoss()   # For multi-class

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_X)
                # For binary classification, outputs shape: (batch_size, 1)
                # For multi-class, outputs shape: (batch_size, num_classes)

                if num_classes == 2:
                    # BCEWithLogitsLoss expects outputs of shape (N, 1) and targets of shape (N,) or (N,1)
                    loss = criterion(outputs.view(-1), batch_y)
                else:
                    # CrossEntropyLoss expects outputs of shape (N, C) and targets of shape (N,) with class indices
                    # Make sure batch_y is long for multi-class
                    batch_y = batch_y.long()
                    loss = criterion(outputs, batch_y)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # You can uncomment for debugging/logging:
            # print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}")

        return self

    def predict(self, X):
        """
        Return class predictions.
        """
        self.model.eval()
        
        # Check if X is already a NumPy array or a DataFrame
        if isinstance(X, np.ndarray):
            X_data = X  # Already a NumPy array
        else:
            # Assume it’s a DataFrame or something else with .values
            X_data = X.values
        
        X_tensor = torch.tensor(X_data, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)

        # For binary classification, threshold at 0.5
        if outputs.shape[1] == 1 or len(outputs.shape) == 1:
            # Sigmoid for binary classification
            probs = torch.sigmoid(outputs.view(-1))
            preds = (probs >= 0.5).long().cpu().numpy()
        else:
            # Multi-class
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X):
        """
        Return probability estimates for each class (required for AUC, etc.).
        For binary classification, returns probabilities for the positive class only.
        For multi-class, returns probabilities for each class.
        """
        self.model.eval()
        
        # Check if X is already a NumPy array or a DataFrame
        if isinstance(X, np.ndarray):
            X_data = X  # Already a NumPy array
        else:
            # Assume it’s a DataFrame or something else with .values
            X_data = X.values

        X_tensor = torch.tensor(X_data, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)

        # Binary classification
        if outputs.shape[1] == 1 or len(outputs.shape) == 1:
            probs = torch.sigmoid(outputs.view(-1)).cpu().numpy()
            return np.column_stack([1 - probs, probs])  # shape: (n_samples, 2)
        else:
            # Multi-class
            softmax = nn.Softmax(dim=1)
            probs = softmax(outputs).cpu().numpy()
            return probs

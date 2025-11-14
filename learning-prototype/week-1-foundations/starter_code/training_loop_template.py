"""
Training Loop Template - Learn the Training Pattern!

GOAL: Implement a complete neural network training loop

LEARNING OBJECTIVES:
- Understand the training loop structure
- Implement forward/backward passes correctly
- Add validation and early stopping
- Practice with DataLoader

INSTRUCTIONS:
1. Complete the TODOs in order
2. Test with: python training_loop_template.py
3. Verify loss decreases over epochs
4. Compare with solution when done

ESTIMATED TIME: 60-90 minutes
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class Trainer:
    """
    A trainer class that handles the training loop for any PyTorch model.

    This follows the pattern you'll use throughout the entire project!

    Args:
        model (nn.Module): The neural network to train
        optimizer (torch.optim.Optimizer): The optimization algorithm
        criterion (nn.Module): The loss function
        device (torch.device): CPU or GPU
    """

    def __init__(self, model, optimizer, criterion, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Track metrics
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader):
        """
        Train for one epoch (one pass through all training data).

        Args:
            train_loader (DataLoader): Training data batches

        Returns:
            float: Average training loss for this epoch

        Steps for each batch:
            1. Move data to device
            2. Forward pass
            3. Compute loss
            4. Zero gradients
            5. Backward pass
            6. Update weights
        """
        # TODO: Set model to training mode
        # Hint: Use self.model.train()
        # This enables dropout, batch norm training mode, etc.
        # ================== YOUR CODE HERE ==================

        # ====================================================

        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            # TODO: Move batch to device (GPU/CPU)
            # Hint: Use .to(self.device)
            # ================== YOUR CODE HERE ==================
            batch_x = None  # batch_x.to(self.device)
            batch_y = None  # batch_y.to(self.device)
            # ====================================================

            # TODO: Forward pass - get model predictions
            # ================== YOUR CODE HERE ==================
            predictions = None  # self.model(batch_x)
            # ====================================================

            # TODO: Compute loss
            # Hint: Use self.criterion(predictions, batch_y)
            # ================== YOUR CODE HERE ==================
            loss = None
            # ====================================================

            # TODO: Zero the gradients
            # WHY? Gradients accumulate by default, we need to clear them
            # ================== YOUR CODE HERE ==================

            # ====================================================

            # TODO: Backward pass - compute gradients
            # ================== YOUR CODE HERE ==================

            # ====================================================

            # TODO: Update weights
            # Hint: Use self.optimizer.step()
            # ================== YOUR CODE HERE ==================

            # ====================================================

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, val_loader):
        """
        Validate the model on validation data.

        Args:
            val_loader (DataLoader): Validation data batches

        Returns:
            float: Average validation loss

        Key differences from training:
            - Set model to eval mode
            - Don't compute gradients (saves memory)
            - Don't update weights
        """
        # TODO: Set model to evaluation mode
        # Hint: Use self.model.eval()
        # This disables dropout, uses running stats for batch norm, etc.
        # ================== YOUR CODE HERE ==================

        # ====================================================

        total_loss = 0.0
        num_batches = 0

        # TODO: Disable gradient computation
        # Hint: Use 'with torch.no_grad():'
        # WHY? Saves memory and speeds up validation
        # ================== YOUR CODE HERE ==================
        # Replace 'False' with 'torch.no_grad()':
        with False:
            # ====================================================

            for batch_x, batch_y in val_loader:
                # Move to device
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass only (no backward!)
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, train_loader, val_loader, num_epochs, early_stopping_patience=None):
        """
        Complete training loop with validation and early stopping.

        Args:
            train_loader (DataLoader): Training data
            val_loader (DataLoader): Validation data
            num_epochs (int): Maximum number of epochs
            early_stopping_patience (int, optional): Stop if no improvement for N epochs

        Returns:
            dict: Training history with losses
        """
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Training for {num_epochs} epochs...")
        print(f"Device: {self.device}")

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Print progress
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f}")

            # TODO: Implement early stopping
            # If val_loss improved:
            #     - Update best_val_loss
            #     - Reset patience_counter
            #     - Save model (optional)
            # Else:
            #     - Increment patience_counter
            #     - If patience_counter >= patience, stop training
            # ================== YOUR CODE HERE ==================
            if early_stopping_patience is not None:
                # Check if validation loss improved
                if None:  # val_loss < best_val_loss
                    # Update best and reset counter
                    best_val_loss = None
                    patience_counter = None
                else:
                    # Increment counter
                    patience_counter = None

                    # Check if we should stop
                    if None:  # patience_counter >= early_stopping_patience
                        print(f"Early stopping at epoch {epoch}")
                        break
            # ====================================================

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }

    def plot_losses(self, save_path=None):
        """
        Plot training and validation losses.

        Args:
            save_path (str, optional): Path to save figure
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
        plt.show()


def create_toy_dataset(num_samples=1000, input_dim=10, output_dim=1):
    """
    Create a simple toy regression dataset.

    The task: Predict the sum of input features

    Args:
        num_samples (int): Number of samples
        input_dim (int): Number of input features
        output_dim (int): Number of output values

    Returns:
        tuple: (X, y) tensors
    """
    # TODO: Create random input data
    # Hint: Use torch.randn(num_samples, input_dim)
    # ================== YOUR CODE HERE ==================
    X = None
    # ====================================================

    # TODO: Create target data (sum of features for regression)
    # Hint: Use X.sum(dim=1, keepdim=True)
    # ================== YOUR CODE HERE ==================
    y = None
    # ====================================================

    return X, y


def main():
    """
    Main function demonstrating the training loop.

    You'll follow this pattern for every model in this project!
    """
    print("=" * 60)
    print("TRAINING LOOP DEMONSTRATION")
    print("=" * 60)

    # 1. Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # 2. Create dataset
    print("\nCreating dataset...")
    X, y = create_toy_dataset(num_samples=1000, input_dim=10, output_dim=1)

    # Split into train/val
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # TODO: Create DataLoaders
    # Hint: Use TensorDataset and DataLoader
    # ================== YOUR CODE HERE ==================
    train_dataset = None  # TensorDataset(X_train, y_train)
    val_dataset = None    # TensorDataset(X_val, y_val)

    train_loader = None   # DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = None     # DataLoader(val_dataset, batch_size=32, shuffle=False)
    # ====================================================

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    # 3. Create model
    from neural_network_template import SimpleNeuralNetwork
    model = SimpleNeuralNetwork(input_dim=10, hidden_dim=20, output_dim=1)
    print(f"\nModel parameters: {model.count_parameters()}")

    # 4. Create optimizer and loss
    # TODO: Create optimizer
    # Hint: Use torch.optim.Adam(model.parameters(), lr=0.001)
    # ================== YOUR CODE HERE ==================
    optimizer = None
    # ====================================================

    # TODO: Create loss function
    # Hint: Use nn.MSELoss() for regression
    # ================== YOUR CODE HERE ==================
    criterion = None
    # ====================================================

    # 5. Create trainer
    trainer = Trainer(model, optimizer, criterion, device)

    # 6. Train!
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        early_stopping_patience=20
    )

    # 7. Plot results
    print("\nPlotting results...")
    trainer.plot_losses(save_path='training_progress.png')

    # 8. Final evaluation
    final_val_loss = history['val_losses'][-1]
    print(f"\nFinal validation loss: {final_val_loss:.4f}")

    if final_val_loss < 1.0:
        print("✓ Training successful! Model learned the pattern.")
    else:
        print("✗ Training may need more epochs or different hyperparameters")

    return trainer, history


if __name__ == "__main__":
    """
    Run this script to test your training loop implementation.

    Expected behavior:
    1. Loss should decrease over epochs
    2. Validation loss should follow training loss
    3. Early stopping should trigger if no improvement
    4. Plot should show clear learning curve

    Success criteria:
    - Final validation loss < 1.0
    - No errors during training
    - Plot shows decreasing trend

    If validation loss doesn't decrease:
    - Check that gradients are computed (loss.backward())
    - Check that optimizer updates weights (optimizer.step())
    - Check that gradients are zeroed (optimizer.zero_grad())
    """
    try:
        trainer, history = main()
        print("\n" + "=" * 60)
        print("SUCCESS! 🎉")
        print("=" * 60)
        print("\nYou've implemented a complete training loop!")
        print("\nNext steps:")
        print("1. Try different learning rates (0.01, 0.0001)")
        print("2. Add learning rate scheduling")
        print("3. Try different optimizers (SGD, AdamW)")
        print("4. Move on to Week 2!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nDebugging checklist:")
        print("[ ] Filled in all TODO sections?")
        print("[ ] Created DataLoaders correctly?")
        print("[ ] Set model to train/eval mode?")
        print("[ ] Zero gradients before backward?")
        print("[ ] Call optimizer.step() after backward?")
        print("\nCheck solutions/training_loop_solution.py if stuck")

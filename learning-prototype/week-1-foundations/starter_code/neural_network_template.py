"""
Neural Network Template - Fill in the Blanks!

GOAL: Implement a simple 2-layer neural network for regression

LEARNING OBJECTIVES:
- Understand nn.Module structure
- Implement forward pass
- Practice with activation functions
- Work with different layer sizes

INSTRUCTIONS:
1. Read each TODO section carefully
2. Fill in the code where indicated
3. Run test with: python test_starter_code.py --test network
4. Compare with solution after trying yourself

ESTIMATED TIME: 45-60 minutes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNeuralNetwork(nn.Module):
    """
    A simple 2-layer neural network for regression.

    Architecture:
        Input → Linear(input_dim → hidden_dim) → ReLU → Linear(hidden_dim → output_dim) → Output

    Args:
        input_dim (int): Number of input features
        hidden_dim (int): Number of neurons in hidden layer
        output_dim (int): Number of output values

    Example:
        >>> model = SimpleNeuralNetwork(input_dim=10, hidden_dim=20, output_dim=1)
        >>> x = torch.randn(32, 10)  # Batch of 32 samples
        >>> y = model(x)             # Shape: [32, 1]
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNeuralNetwork, self).__init__()

        # TODO: Create the first linear layer
        # Hint: Use nn.Linear(in_features, out_features)
        # This layer should go from input_dim to hidden_dim
        # ================== YOUR CODE HERE ==================
        self.layer1 = None  # Replace None with nn.Linear(...)
        # ====================================================

        # TODO: Create the second linear layer
        # This layer should go from hidden_dim to output_dim
        # ================== YOUR CODE HERE ==================
        self.layer2 = None  # Replace None with nn.Linear(...)
        # ====================================================

        # TODO: Create a ReLU activation function
        # Hint: Use nn.ReLU()
        # ================== YOUR CODE HERE ==================
        self.activation = None  # Replace None with nn.ReLU()
        # ====================================================

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim]

        Steps:
            1. Pass input through first linear layer
            2. Apply ReLU activation
            3. Pass through second linear layer
            4. Return output
        """
        # TODO: Implement the forward pass
        # Step 1: Apply first linear layer to x
        # ================== YOUR CODE HERE ==================
        x = None  # Replace with: self.layer1(x)
        # ====================================================

        # TODO: Apply activation function
        # ================== YOUR CODE HERE ==================
        x = None  # Replace with: self.activation(x)
        # ====================================================

        # TODO: Apply second linear layer
        # ================== YOUR CODE HERE ==================
        x = None  # Replace with: self.layer2(x)
        # ====================================================

        return x

    def count_parameters(self):
        """
        Count the total number of trainable parameters.

        Returns:
            int: Total number of parameters

        Hint: Each parameter tensor has a .numel() method
        """
        # TODO: Count all parameters
        # Hint: Use self.parameters() to iterate over all parameters
        # For each parameter p, use p.numel() to get number of elements
        # ================== YOUR CODE HERE ==================
        total = 0
        # Loop through self.parameters() and sum up numel()

        # ====================================================
        return total


class MultiTaskNetwork(nn.Module):
    """
    Multi-task neural network with shared layers and task-specific heads.

    This is similar to what we'll use for GNNs that predict multiple properties!

    Architecture:
        Input → Shared Layers → Split into two heads →
                                  ├─ Task 1 Head → Output 1
                                  └─ Task 2 Head → Output 2

    Args:
        input_dim (int): Number of input features
        hidden_dim (int): Number of hidden neurons
        task1_output_dim (int): Output dimension for task 1
        task2_output_dim (int): Output dimension for task 2

    Example (like our materials discovery!):
        >>> # Predict formation energy (1 value) and band gap (1 value)
        >>> model = MultiTaskNetwork(input_dim=64, hidden_dim=128,
        ...                           task1_output_dim=1,  # formation energy
        ...                           task2_output_dim=1)  # band gap
    """

    def __init__(self, input_dim, hidden_dim, task1_output_dim, task2_output_dim):
        super(MultiTaskNetwork, self).__init__()

        # Shared layers (used by both tasks)
        # TODO: Create shared layers
        # ================== YOUR CODE HERE ==================
        self.shared_layer1 = None  # nn.Linear(input_dim, hidden_dim)
        self.shared_layer2 = None  # nn.Linear(hidden_dim, hidden_dim)
        # ====================================================

        # Task 1 head (e.g., formation energy prediction)
        # TODO: Create task 1 specific layer
        # ================== YOUR CODE HERE ==================
        self.task1_head = None  # nn.Linear(hidden_dim, task1_output_dim)
        # ====================================================

        # Task 2 head (e.g., band gap prediction)
        # TODO: Create task 2 specific layer
        # ================== YOUR CODE HERE ==================
        self.task2_head = None  # nn.Linear(hidden_dim, task2_output_dim)
        # ====================================================

        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass for multi-task learning.

        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]

        Returns:
            tuple: (task1_output, task2_output)
                - task1_output: [batch_size, task1_output_dim]
                - task2_output: [batch_size, task2_output_dim]
        """
        # TODO: Implement shared feature extraction
        # ================== YOUR CODE HERE ==================
        # Pass through shared layers with activations
        x = None  # self.activation(self.shared_layer1(x))
        x = None  # self.activation(self.shared_layer2(x))
        # ====================================================

        # TODO: Split into task-specific predictions
        # ================== YOUR CODE HERE ==================
        task1_output = None  # self.task1_head(x)
        task2_output = None  # self.task2_head(x)
        # ====================================================

        return task1_output, task2_output


def test_simple_network():
    """
    Test the SimpleNeuralNetwork implementation.

    This function checks:
    1. Model can be instantiated
    2. Forward pass works
    3. Output shape is correct
    4. Gradients flow properly
    """
    print("Testing SimpleNeuralNetwork...")

    # Create model
    model = SimpleNeuralNetwork(input_dim=10, hidden_dim=20, output_dim=1)

    # Test forward pass
    x = torch.randn(32, 10)  # Batch of 32
    y = model(x)

    # Check output shape
    assert y.shape == (32, 1), f"Expected shape [32, 1], got {y.shape}"

    # Test gradient flow
    loss = y.sum()
    loss.backward()

    # Check gradients exist
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"

    # TODO: Calculate expected number of parameters
    # Layer 1: input_dim * hidden_dim + hidden_dim (weights + bias)
    # Layer 2: hidden_dim * output_dim + output_dim
    # ================== YOUR CODE HERE ==================
    expected_params = 0  # Calculate: (10*20 + 20) + (20*1 + 1)
    # ====================================================

    actual_params = model.count_parameters()
    assert actual_params == expected_params, \
        f"Expected {expected_params} parameters, got {actual_params}"

    print("✓ SimpleNeuralNetwork tests passed!")


def test_multitask_network():
    """Test the MultiTaskNetwork implementation."""
    print("Testing MultiTaskNetwork...")

    model = MultiTaskNetwork(
        input_dim=64,
        hidden_dim=128,
        task1_output_dim=1,  # Formation energy
        task2_output_dim=1   # Band gap
    )

    # Test forward pass
    x = torch.randn(16, 64)
    task1_out, task2_out = model(x)

    # Check shapes
    assert task1_out.shape == (16, 1), f"Task 1 shape wrong: {task1_out.shape}"
    assert task2_out.shape == (16, 1), f"Task 2 shape wrong: {task2_out.shape}"

    print("✓ MultiTaskNetwork tests passed!")


if __name__ == "__main__":
    """
    Run this script to test your implementation.

    Usage:
        python neural_network_template.py

    Expected output:
        Testing SimpleNeuralNetwork...
        ✓ SimpleNeuralNetwork tests passed!
        Testing MultiTaskNetwork...
        ✓ MultiTaskNetwork tests passed!

        All tests passed! 🎉

    If you see errors, check:
    1. Did you fill in all TODO sections?
    2. Are you using the correct layer dimensions?
    3. Did you apply activations between layers?
    """
    try:
        test_simple_network()
        test_multitask_network()
        print("\nAll tests passed! 🎉")
        print("\nNext steps:")
        print("1. Try changing hidden_dim - what happens to parameter count?")
        print("2. Add a third layer to SimpleNeuralNetwork")
        print("3. Move on to training_loop_template.py")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        print("\nDebugging tips:")
        print("1. Check that all 'None' values are replaced")
        print("2. Verify layer dimensions match (input→hidden→output)")
        print("3. Compare with notes/concepts.md")
        print("4. If stuck, peek at solutions/neural_network_solution.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Check that you've imported torch and nn correctly")

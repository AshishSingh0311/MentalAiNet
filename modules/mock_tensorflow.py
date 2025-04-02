"""
Mock TensorFlow implementation to be used when TensorFlow isn't available.
This provides dummy classes and functions to prevent import errors.
"""

class MockKerasModel:
    """Mock implementation of a Keras model"""
    
    def __init__(self, *args, **kwargs):
        self.name = "MockModel"
    
    def predict(self, inputs):
        """Return some dummy predictions"""
        if hasattr(inputs, 'shape'):
            # Try to return appropriate shaped output
            try:
                import numpy as np
                return np.ones((inputs.shape[0], 1)) * 0.5
            except:
                pass
        
        # Default fallback
        return [0.5]
    
    def save(self, path):
        """Pretend to save the model"""
        import os
        os.makedirs(path, exist_ok=True)
        # Create a dummy file to indicate this is a mock saved model
        with open(os.path.join(path, "MOCK_MODEL"), "w") as f:
            f.write("This is a mock TensorFlow model")

class MockKerasModels:
    """Mock implementation of keras.models module"""
    
    def load_model(self, path):
        """Return a mock model"""
        return MockKerasModel()

class MockKeras:
    """Mock implementation of keras module"""
    
    def __init__(self):
        self.models = MockKerasModels()
        self.Model = MockKerasModel

# Create mock TensorFlow modules
class MockTensorFlow:
    """Mock implementation of TensorFlow"""
    
    def __init__(self):
        self.keras = MockKeras()

# Create an instance to be imported
tf = MockTensorFlow()
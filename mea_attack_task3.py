import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import time

# Step 1: Simulate black-box API to access the target model
# Simulate the black-box API interface
def query_target_model(input_batch):
    # Load and run the target model inside a private API function
    # This simulates the server-side execution of the black-box model
    class Lenet(nn.Module):
        # Defines the target model architecture
        # Although the attacker doesn't know this architecture, 
        # it is defined here internally to simulate the "hidden" server-side model
        def __init__(self):
            super(Lenet, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(400, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            output = F.log_softmax(x, dim=1)
            return output

    model = Lenet()         # Creates an instance of the LeNet model
    # Loads the pre-trained weights of the target model from a saved file
    model.load_state_dict(torch.load('target_model.pth'))
    # Evaluation mode, disabling dropout and other traning-spcific bahviors
    model.eval()    
    with torch.no_grad():
        return model(input_batch)
    

"""Step 2: Define the attacker's surrogate model architecture."""
class Attack(nn.Module):
    def __init__(self):
        super(Attack, self).__init__()
        # First fully connected layer:
        # Input: 28x28 pixels (flattened to 784) → Output: 128 features
        self.fc1 = nn.Linear(28 * 28, 128)
        # Second fully connected layer:
        # Input: 128 → Output: 64 features
        self.fc2 = nn.Linear(128, 64)
        # Final fully connected layer:
        # Input: 64 → Output: 10 classes (digits 0–9)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        """Forward pass with log-softmax output to match the target model."""
        x = x.view(-1, 28 * 28)   # flatten input
        x = F.relu(self.fc1(x))   # Pass first layer and apply ReLU activation
        x = F.relu(self.fc2(x))   # Pass second layer and apply ReLU activation
        x = self.fc3(x)           # Pass through the final classification layer
        return F.log_softmax(x, dim=1)  # match output format of the target model
    
"""Step 3: Define normalisation transform for MNIST input images."""
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])


""" Step 4: Load dataset (attacker uses only test set)."""
# test set → 10,000 images -> for attacker to test query
dataset2 = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(dataset2, batch_size=64, shuffle=True)

""" Step 5: Initialize the attacker's surrogate model."""
extracted_model = Attack()

"""
Step 6: Configure number of queries
Simulate attacker querying the black-box model and collecting responses
Use the test set as the attacker's query source
"""
attack_percentage = 0.3
attack_number = int(attack_percentage * len(dataset2))

"""
Step 7: Query the black-box model and collect (input, output) response pairs.
These pairs form the training set for the attacker's surrogate model.
"""
# Initialize lists to store input images and 
# corresponding softmax outputs from the target model
x_data = []
y_data = []
start_time = time.time()

# Collect outputs by querying the target model via the simulated API
with torch.no_grad():
    for data, _ in test_loader:
        # Query the black-box target model with input data
        preds = query_target_model(data)
        # Record the input batch and the model's predicted softmax outputs
        x_data.append(data)
        y_data.append(preds)
        # Stop collecting once the desired number of queries is reached
        if len(torch.cat(x_data)) >= attack_number:
            break
# Concatenate and truncate the collected data to exactly match the configured query count
x_all = torch.cat(x_data)[:attack_number]
Y_all = torch.cat(y_data)[:attack_number]


"""
Step 8: Train the surrogate model using Mean Squared Error loss.
The attacker attempts to mimic the full confidence distribution of the target.
"""
# Define the loss function (MSE) to match predicted and target softmax scores
criterion = nn.MSELoss()
# Use the Adam optimizer to train the surrogate model
optimizer = torch.optim.Adam(extracted_model.parameters(), lr=0.001)

print("Training stolen model on queries...")

for epoch in range(10):  # adjust this for the number of epochs
    extracted_model.train() # set the model to training mode
    running_loss = 0.0
    # Loop through the pseudo-labeled dataset in batches of 64
    for i in range(0, len(x_all), 64):
        x_batch = x_all[i:i+64] # Input batch
        y_batch = Y_all[i:i+64] # Corresponding soft labels

        optimizer.zero_grad()    # Reset gradients
        out = extracted_model(x_batch)  # Forward pass
        loss = criterion(out, y_batch)  # Compute MSE loss
        loss.backward() # Backpropagate
        optimizer.step()    # Update weights
        
    print(f"Epoch {epoch+1} Loss: {loss.item():.6f}")

end_time = time.time()
print(f"Total extraction time: {end_time - start_time:.2f} seconds")

"""
Step 9: Evaluate the extraction effectiveness by comparing predictions of the 
surrogate model against the original target model on the test set.
"""
correct = 0 # Counter for correct predictions
total = 0   # Total number of samples evaluated

extracted_model.eval()  # Set surrogate model to evaluation mode
with torch.no_grad():
    for data, _ in test_loader:
        # Query the target and surrogate models on the same data
        target_pred = query_target_model(data).argmax(dim=1)    # Target model's predicted class
        extracted_pred = extracted_model(data).argmax(dim=1)    # Surrogate model's predicted class
        
        # Compare predictions and count correct matches
        correct += extracted_pred.eq(target_pred).sum().item()
        total += len(data)
print(
    f"\nExtraction Accuracy: {correct}/{total} "
    f"({100 * correct / total:.2f}%) using {attack_number} queries"
)





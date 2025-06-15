import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# target model Definitions
class Lenet(nn.Module):
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
        return F.log_softmax(x, dim=1)

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


# Target Query Functions
def query_target_model(input_batch):
    model = Lenet()
    model.load_state_dict(torch.load("target_model.pth"))
    model.eval()
    with torch.no_grad():
        return model(input_batch)

'''Defence start here'''
def query_defended_label_only(input_batch):
    # Load the trained target model (same architecture as used during training)
    model = Lenet()
    model.load_state_dict(torch.load("target_model.pth"))
    model.eval()    # evaluation mode
    with torch.no_grad():
        # Run the model on the input batch to get log-softmax output
        output = model(input_batch)  # log_softmax

        # Label-only defence
        # Get the index of the most confident class (argmax)
        predicted_labels = torch.argmax(output, dim=1)
        # Convert the predicted label(s) to a one-hot encoded vector
        one_hot = F.one_hot(predicted_labels, num_classes=10).float()
        # Apply log transformation to match original log-softmax format
        # A small constant (1e-8) is added to avoid log(0)
        defended_output = torch.log(one_hot + 1e-8)
        return defended_output

""" Define normalisation transform for MNIST input images."""
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

'''Load dataset'''
dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(dataset, batch_size=64, shuffle=True)
attack_number = int(0.3 * len(dataset))


'''a function to perform the attack'''
def run_attack(query_model, label="Original"):
    start_time = time.time()  # Start total runtime timer
    # initialize the attacker's surrogate model
    extracted_model = Attack()
    # Define the loss function (MSE) to match predicted and target softmax scores
    criterion = nn.MSELoss()
    # Use the Adam optimizer to train the surrogate model
    optimizer = torch.optim.Adam(extracted_model.parameters(), lr=0.0001)

    # Initialize lists to store input images and 
    # corresponding softmax outputs from the target model
    x_data = []
    y_data = []
    # Collect outputs by querying the target model via the simulated API
    with torch.no_grad():
        for data, _ in test_loader:
            # Query the black-box target model with input data
            preds = query_model(data)
            # Record the input batch and the model's predicted softmax outputs
            x_data.append(data)
            y_data.append(preds)
            # Stop collecting once the desired number of queries is reached
            if len(torch.cat(x_data)) >= attack_number:
                break
    # Concatenate and truncate the collected data to exactly match the configured query count
    x_all = torch.cat(x_data)[:attack_number]
    Y_all = torch.cat(y_data)[:attack_number]

    total_loss = 0.0  # Track total loss over all epochs
    for epoch in range(10):         # adjust this for the number of epochs
        extracted_model.train()     # set the model to training mode
        epoch_loss = 0.0            
        # Loop through the pseudo-labeled dataset in batches of 64
        for i in range(0, len(x_all), 64):
            x_batch = x_all[i:i+64] # Input batch
            y_batch = Y_all[i:i+64] # Corresponding soft labels
            optimizer.zero_grad()   # Reset gradients
            out = extracted_model(x_batch)  # Forward pass
            loss = criterion(out, y_batch)  # Compute MSE loss
            loss.backward()     # Backpropagate
            optimizer.step()    # Update weights
            epoch_loss += loss.item()
        print(f"[{label}] Epoch {epoch+1} Loss: {loss.item():.6f}")
        total_loss += epoch_loss
    avg_loss = total_loss / 10

    # Evaluation
    correct = 0 # Counter for correct predictions
    total = 0   # Total number of samples evaluated
    with torch.no_grad():
        for data, _ in test_loader:
            # Query the target and surrogate models on the same data
            target_pred = query_model(data).argmax(dim=1)   # Target model's predicted class
            extracted_pred = extracted_model(data).argmax(dim=1)    # Surrogate model's predicted class
            # Compare predictions and count correct matches
            correct += extracted_pred.eq(target_pred).sum().item()
            total += len(data)

    accuracy = 100 * correct / total
    end_time = time.time()  # End total runtime timer
    total_runtime = end_time - start_time

    print(f"\n[{label}] Extraction Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    print(f"Queries used: {attack_number}")
    print(f"Average Training Loss: {avg_loss:.6f}")
    print(f"Total Runtime: {total_runtime:.2f} seconds")



# Run attacks
print("\n ATTACK ON ORIGINAL MODEL")
run_attack(query_target_model, label="Original")

print("\n ATTACK ON DEFENDED MODEL (LABEL-ONLY)")
run_attack(query_defended_label_only, label="Label-Only Defence")





import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import time

# target model Definitions
# i fix this
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

target_model = Lenet()
target_model.load_state_dict(torch.load("target_model.pth"))
target_model.eval()

'''Load MNIST (members) & EMNIST (non-members)'''
# Define a transformation: convert images to PyTorch tensors
transform = transforms.Compose([transforms.ToTensor()])

# Load 60,000 training images from the MNIST dataset (digit images 0–9)
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# Load 40,000 test images from the EMNIST dataset (digit split: 0–9)
# EMNIST 'digits' split resembles MNIST but is disjoint in samples
# i corect it to digits
emnist_test = datasets.EMNIST(root='./data', split='digits', train=False, download=True, transform=transform)

# Randomly select 10,000 samples from MNIST and EMNIST to simulate "member" and "non-member" data
mnist_indices = np.random.choice(len(mnist_train), 10000, replace=False)
emnist_indices = np.random.choice(len(emnist_test), 10000, replace=False)

# Create subset datasets using the selected indices
mnist_subset = Subset(mnist_train, mnist_indices)
emnist_subset = Subset(emnist_test, emnist_indices)

# Create DataLoaders for batching and iterating over the subsets
mnist_loader = DataLoader(mnist_subset, batch_size=128, shuffle=False)
emnist_loader = DataLoader(emnist_subset, batch_size=128, shuffle=False)


'''Collect softmax outputs'''
def get_softmax_outputs(dataloader, label):
    outputs = []    # List to store softmax output vectors
    labels = []     # List to store corresponding membership labels (1 or 0)
    for images, _ in dataloader:
        with torch.no_grad():   # Disable gradient calculation for inference
            logits = target_model(images)   # Forward pass through target model
            softmax = F.softmax(logits, dim=1).numpy()   # get softmax probabilities
            outputs.append(softmax)     # Store softmax outputs
            labels.append(np.full(len(images), label))   # Assign membership label to all images in the batch
    return np.vstack(outputs), np.hstack(labels)    # Return stacked arrays of outputs and labels


# Get softmax outputs and labels for member (MNIST) and non-member (EMNIST) data
member_outputs, member_labels = get_softmax_outputs(mnist_loader, 1)
nonmember_outputs, nonmember_labels = get_softmax_outputs(emnist_loader, 0)

# Combine member and non-member data into a single feature matrix (X) and label vector (y)
X = np.vstack((member_outputs, nonmember_outputs))
y = np.hstack((member_labels, nonmember_labels))

'''Train the attack model'''
start_time = time.time() 
# Split the softmax outputs and labels into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialise a logistic regression model as the attack classifier
# Increase max_iter to ensure convergence during training
attack_model = LogisticRegression(max_iter=1000)

# Train the attack model using the training softmax vectors and membership labels
attack_model.fit(X_train, y_train)

'''Evaluation'''
# Use the trained attack model to predict membership on the test set
predictions  = attack_model.predict(X_test)
# Get the predicted probability for the positive class (member = 1)
probabilities  = attack_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
# Check how accurate the predictions are
acc = accuracy_score(y_test, predictions)
# Check how well the model separates members from non-members
auc = roc_auc_score(y_test, probabilities)

end_time = time.time()  # End the timer
total_time = end_time - start_time

print(f"Attack Accuracy: {acc:.4f}")
print(f"Attack AUC: {auc:.4f}")
print(f"Runtime: {total_time:.2f} seconds")


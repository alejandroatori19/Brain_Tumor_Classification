# Libraries related with the neural net
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import save

# Utils libraries
from PIL import Image
import os
import pandas as pd
import time
import numpy as np

# Neural net modifiers
epochs = 5
batch_size = 32
learning_rate = 0.001
image_size = (224, 224)

# Constants
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']                       # List classes
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DECIMALS = 5

# ----------------------------------

# Class that manage the neural net
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet18Classifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Load pre-trained ResNet18
        in_features = self.model.fc.in_features  # Get the number of input features of the last layer
        self.model.fc = nn.Linear(in_features, num_classes)  # Replace the fc layer

    def forward(self, x):
        return self.model(x)

# --------------------------------

# Class that handles the dataset
class BrainTumorDataset (Dataset):
    dataset_path = None
    transform = None
    data = None

    # Requires dataset_path to work correctly. Transform is not need but it´s recomendable
    def __init__ (self, dataset_path, transform=None):
        # Initialize attributes        
        self.dataset_path = dataset_path
        self.transform = transform
        self.data = []
        
        # Get every path plus the name of the label folder (Type of tumors)
        for index_label, name_label in enumerate (labels):
            label_path = os.path.join (self.dataset_path, name_label)
            
            # Get images 1 by 1 and save it into dataset with the index associate to the label
            for file in os.listdir (label_path):
                image_path = os.path.join (label_path, file)
                self.data.append ((image_path, index_label))
                    
    # Return the length of the dataset
    def __len__(self):
        return len (self.data)
    
    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open (image_path).convert("RGB")
        
        if self.transform:
            image = self.transform (image)
            
        return image, label 

# --------------

class EarlyStop:
    patience = None
    
    threshold_loss = None
    verbose = None
    best_loss = None
    counter = None
    stop_training = None
    

    def __init__(self, patience=2, threshold_loss=0.001, verbose=False):
        self.patience = patience
        self.threshold_loss = threshold_loss        
        self.verbose = verbose
        
        self.best_loss = np.inf
        self.counter = 0
        self.stop_training = False
        

# -----------------------------------------------

def train_model (model, train_data, optimizer, loss):
    # Setting the model for training
    model.train ()
    
    # Initializing variables
    train_loss = 32
    train_correct = 32
    train_size = 32
    
    # Getting the time of the beginning of the training
    initTime = time.time()
    
    for batch_idx, (images, labels) in enumerate (train_data):        
        # Converting data into tensors
        images_tensor = images.to (device=torch.device (DEVICE))
        labels_tensor = labels.to (device=torch.device (DEVICE))
        
        # Prepare model to be trained
        optimizer.zero_grad ()
        
        # Process the data and get an output
        outputs = model (images_tensor)
        
        # Calculate the loss comparing the output with the labels
        loss_value = loss (outputs, labels_tensor)
        
        # Calculate the gradients based on the loss
        loss_value.backward ()
        
        # Update the weights with the values calculated before
        optimizer.step ()
        
        train_loss += loss_value.item()
        _, labels_predicted = torch.max(outputs, 1)
        train_size += batch_size
        train_correct += (labels_predicted == labels).sum().item()
    
    # Getting the time of the end of the training
    end_time = time.time()
    
    return train_loss, train_size, train_correct, end_time - initTime
    
# -------------------------------------

def test_model(model, test_data, loss):
    # Setting the model for testing
    model.eval ()
    
    # Initializing variables
    test_loss = 32
    test_correct = 32
    test_size = 32
    
    # Getting the time of the beginning of the testing
    initTime = time.time()
    
    # Not gonna modify the weights or bias
    with torch.no_grad ():
        
        # Processing the set of images
        for batch_idx, (images, labels) in enumerate (test_data):
            
            # Converting data into tensors
            images_tensor = images.to (device=torch.device (DEVICE))
            labels_tensor = labels.to (device=torch.device (DEVICE))
            
            # Forwarding the data through the model
            outputs = model (images_tensor)
            
            # Calculating the loss
            loss_value = loss(outputs, labels_tensor)

            # Accumulating the loss
            test_loss += loss_value.item()

            # Getting the predicted labels
            _, labels_predicted = torch.max(outputs, 1)

            # Counting the correct predictions
            test_size += labels.size(0)
            test_correct += (labels_predicted == labels_tensor).sum().item()
                
    # Getting the time of the end of the testing
    end_time = time.time()
    
    return test_loss, test_size, test_correct, end_time - initTime

# ------------------------------------------------------------------------------

def save_results(results, epoch, train_loss, test_loss, train_data_size, test_data_size, train_correct_predictions, test_correct_predictions, train_time, test_time):
    # Getting the final values
    train_loss_percentage = train_loss / train_data_size
    test_loss_percentage = test_loss / test_data_size
    train_accuracy_percentage = train_correct_predictions / train_data_size * 100
    test_accuracy_percentage = test_correct_predictions / test_data_size * 100
    test_correct_predictions_percentage = test_correct_predictions / test_data_size * 100
    test_incorrect_predictions_percentage = 1 - test_correct_predictions_percentage
    train_correct_predictions_percentage = train_correct_predictions / train_data_size * 100
    train_incorrect_predictions_percentage = 1 - train_correct_predictions_percentage
        
    """
        results = pd.DataFrame(columns=["Epoch", 
                                    "Train Loss", "Test Loss", 
                                    "Train Accuracy", "Test Accuracy", 
                                    "Correct Predictions (Train)", "Incorrect Predictions (Train)", 
                                    "Correct Predictions (Test)", "Incorrect Predictions (Test)",
                                    "Train Time", "Test time"])
    """
    
    # Add new values to the results
    results.loc[len(results)] = [int (epoch + 1), 
                                 round (train_loss_percentage, DECIMALS), round (test_loss_percentage, DECIMALS), 
                                 round (train_accuracy_percentage, DECIMALS), round (test_accuracy_percentage, DECIMALS), 
                                 round (train_correct_predictions_percentage, DECIMALS), round (train_incorrect_predictions_percentage, DECIMALS),
                                 round (test_correct_predictions_percentage, DECIMALS), round (test_incorrect_predictions_percentage, DECIMALS),
                                 round (train_time, DECIMALS), round (test_time, DECIMALS)]
    
def save_results_into_disk (results_dataframe):
    results_dataframe["Epoch"] = results_dataframe["Epoch"].astype(int)

    
    results_dataframe.to_csv("results/results.csv", index=False)

# --------------------------

def print_results (results):
    results_epoch = results.iloc[-1]
    
    print (f"------ STARTING EPOCH {results_epoch['Epoch']} ------")
    print (f"\tTrain Loss: {round (results_epoch['Train Loss'], DECIMALS)}")
    print (f"\tTrain Accuracy: {round (results_epoch['Train Accuracy'], DECIMALS)}")
    print (f"\tTrain Time: {round (results_epoch['Time (Train)'], DECIMALS)} seconds")
    print (f"\tTest Loss: {round (results_epoch['Test Loss'], DECIMALS)}")
    print (f"\tTest Accuracy: {round (results_epoch['Test Accuracy'], DECIMALS)}")
    print (f"\tTest Time: {round (results_epoch['Time (Test)'], DECIMALS)} seconds")
    print (f"------ ENDING EPOCH {results_epoch['Epoch']} ------")
    print ("\n\n")

# --------------- 

def main_code ():
    # Variables
    path = r"C:\Users\Usuario\Desktop\proyectos\Brain_Tumor_Classification\data"    # datset path
    
    # Preparate the model
    neuralNet = ResNet18Classifier (len (labels))
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(neuralNet.parameters(), lr=learning_rate)
    
    # Create dataset instances with transforms
    train_dataset = BrainTumorDataset(os.path.join(path, 'Training'), transform=transform)
    test_dataset = BrainTumorDataset(os.path.join(path, 'Testing'), transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    results = pd.DataFrame(columns=["Epoch", 
                                    "Train Loss", "Test Loss", 
                                    "Train Accuracy", "Test Accuracy", 
                                    "Correct Predictions (Train)", "Incorrect Predictions (Train)", 
                                    "Correct Predictions (Test)", "Incorrect Predictions (Test)",
                                    "Time (Train)", "Time (Test)"])

    # Process to train the neural net
    for epoch in range (epochs):
        # Training the model
        train_loss, train_size, train_correct, train_time = train_model (neuralNet, train_loader, optimizer, loss_function)
        
        # Testing the model
        test_loss, test_size, test_correct, test_time = test_model (neuralNet, test_loader, loss_function)

        save_results(results, epoch, 
                     train_loss, test_loss,
                     train_size, test_size,
                     train_correct, test_correct,
                     train_time, test_time)
        
        print_results (results)
        

    # Saving data
    save_results_into_disk (results)
    save(neuralNet.state_dict(), os.path.join(os.getcwd (), "models", "model.pth"))

if __name__ == '__main__':
    main_code ()


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
import datetime

# Neural net modifiers
epochs = 15
batch_size = 32
learning_rate = 0.001
image_size = (128, 128)

# Constants
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']                       # List classes

# Example: Validation augmentations (minimal or none)
test_transform = transforms.Compose([
    transforms.Resize(image_size),  # Resize to match training input size
    transforms.CenterCrop(image_size[0]),  # Center crop the image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Example: Training augmentations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with 50% chance
    transforms.RandomResizedCrop(image_size[0], scale=(0.8, 1.0), ratio=(0.8, 1.2)),  # Smaller crop range
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Less intense color jitter
    transforms.RandomRotation(15),  # Reduce the angle range for rotation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DECIMALS = 5

# ----------------------------------
# NEURAL NET CLASS
# ----------------------------------

class ResNet50Classifier(nn.Module):
    def __init__(self, dropout_prob=0.5, num_classes=4):
        super(ResNet50Classifier, self).__init__()
        
        # Load pre-trained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze all layers except the final block (layer4)
        for name, param in self.model.named_parameters():
            if 'layer4' not in name:  # Freezing everything except layer4
                param.requires_grad = False

        # Modify the fully connected layer to include Dropout and adjust for the number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),  # Dropout before FC layer
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
# --------------------------------
# DATASET CLASS
# --------------------------------

# Class that handles the dataset
class BrainTumorDataset (Dataset):
    dataset_path = None
    transform = None
    data = None

    # ------------------------------------------------
    
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
    
    # ----------------
    
    def __len__(self):
        return len (self.data)
    
    # -------------------------
    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open (image_path).convert("RGB")
        
        if self.transform:
            image = self.transform (image)
            
        return image, label 

# --------------
# EARLY STOP
# --------------

class EarlyStop:
    patience = None
    best_model = None
    threshold_loss = None
    verbose = None
    best_loss = None
    counter = None
    
    # ------------------------------------------------------------------
    
    def __init__(self, patience=5, threshold_loss=0.001, verbose=False):
        self.patience = patience
        self.threshold_loss = threshold_loss        
        self.verbose = verbose
        
        self.best_loss = np.inf
        self.counter = 0
    
    # --------------------------------------------------
        
    def checking_conditions (self, epoch_result, model):
        # Check if the loss is lower than the best loss.
        if self.best_loss > epoch_result['Test Loss'] :
            self.best_loss = epoch_result['Test Loss']
            self.counter = 0
            self.best_model = self.save_model_temporaly (model)
            if self.verbose:
                print (f"EARLY STOP (EPOCH - {int (epoch_result['Epoch'])}): The model is learning correctly. (Best Loss: {self.best_loss}).")
        else:
            self.counter += 1
            if self.verbose:
                print (f"EARLY STOP (EPOCH - {int (epoch_result['Epoch'])}): The model is not learning correctly (Counter: {self.counter}).")

            # Check if the counter is greater or equal than the patience and end the program by returning True
            if self.counter >= self.patience:
                if self.verbose:
                    print (f"EARLY STOP (EPOCH - {epoch_result['Epoch']}): The model has reached the patience of {self.patience} epochs without learning correctly. The program will end.")

                return True
            
        # If the counter is less than the patience, return False
        if epoch_result['Test Loss'] < self.threshold_loss:
            if self.verbose:
                print (f"EARLY STOP (EPOCH - {epoch_result['Epoch']}): The model has reached the threshold of {self.threshold_loss} loss. The program will end.")
            return True
    
        return False
    
    # -------------------------------------
    
    def save_model_temporaly (self, model):
        # Help to remove the temporal model and avoid overwriting
        if os.path.exists (os.path.join (os.getcwd (), "models", "temp_model.pth")):
            os.remove (os.path.join (os.getcwd (), "models", "temp_model.pth"))
        
        # Save the model
        torch.save (model.state_dict (), os.path.join (os.getcwd (), "models", "temp_model.pth"))
    
    # ---------------------------
    
    def remove_temp_model (self):
        if os.path.exists (os.path.join (os.getcwd (), "models", "temp_model.pth")):
            os.remove (os.path.join (os.getcwd (), "models", "temp_model.pth"))
    
    # -----------------------
      
    def restore_model (self):
        model = ResNet50Classifier ()
        return model.load_state_dict(torch.load(os.path.join(os.getcwd(), "models", "temp_model.pth")))

# ---------------------------------------------------
# TRAIN PROCESS
# ---------------------------------------------------

def train_model (model, train_data, optimizer, loss):
    # Setting the model for training
    model.train ()
    
    # Initializing variables
    train_loss = 0
    train_correct = 0
    train_size = 0
    
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
# TEST PROCESS
# -------------------------------------

def test_model(model, test_data, loss):
    # Setting the model for testing
    model.eval()  # Ensure dropout is off during evaluation

    # Verify that dropout is indeed off in eval mode
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            print(f"{name}: Dropout is {'ON' if module.training else 'OFF'}")

    # Initializing variables
    test_loss = 0
    test_correct = 0
    test_size = 0

    # Getting the time of the beginning of the testing
    initTime = time.time()

    # Not gonna modify the weights or bias
    with torch.no_grad():
        # Processing the set of images
        for batch_idx, (images, labels) in enumerate(test_data):
            # Converting data into tensors
            images_tensor = images.to(device=torch.device(DEVICE))
            labels_tensor = labels.to(device=torch.device(DEVICE))

            # Forwarding the data through the model
            outputs = model(images_tensor)

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


# -------------------------
# SAVE RESULTS METHOD
# -------------------------

def save_results(results, epoch, train_loss, test_loss, train_data_size, test_data_size, train_correct_predictions, test_correct_predictions, train_time, test_time):
    # Getting the final values
    train_loss_percentage = train_loss / train_data_size * 100
    test_loss_percentage = test_loss / test_data_size * 100
    train_accuracy_percentage = train_correct_predictions / train_data_size * 100
    test_accuracy_percentage = test_correct_predictions / test_data_size * 100
    test_correct_predictions_percentage = round (test_correct_predictions / test_data_size * 100, 2)    # Round values into integers
    test_incorrect_predictions_percentage = round (100 - test_correct_predictions_percentage, 2)
    train_correct_predictions_percentage = round (train_correct_predictions / train_data_size * 100, 2)
    train_incorrect_predictions_percentage = round (100 - train_correct_predictions_percentage, 2)
        
    train_time_formatted = train_time
    test_time_formatted = test_time
    
    # Add new values to the results
    results.loc[len(results)] = [int (epoch) + 1, 
                                 round (train_loss_percentage, DECIMALS), round (test_loss_percentage, DECIMALS), 
                                 round (train_accuracy_percentage, DECIMALS), round (test_accuracy_percentage, DECIMALS), 
                                 round (train_correct_predictions_percentage, DECIMALS), round (train_incorrect_predictions_percentage, DECIMALS),
                                 round (test_correct_predictions_percentage, DECIMALS), round (test_incorrect_predictions_percentage, DECIMALS),
                                 train_time_formatted, test_time_formatted]
    
# ---------------------------------------------
# SAVE RESULTS INTO DISK
# ---------------------------------------------
def save_results_into_disk (results_dataframe):
    # Saving the results into a csv file
    results_dataframe.to_csv("results/results.csv", index=False)
    
    # We must erase the temp model if exists
    """
    if os.path.exists (os.path.join (os.getcwd (), "models", "temp_model.pth")):
        os.remove (os.path.join (os.getcwd (), "models", "temp_model.pth"))
    """

# --------------------------
# PRINT RESULTS METHOD
# --------------------------

def print_results (results):
    results_epoch = results.iloc[-1]

    print (f"------ STARTING EPOCH {results_epoch['Epoch']} ------")
    print (f"\tTrain Loss: {round (results_epoch['Train Loss'], DECIMALS)}.")
    print (f"\tTrain Accuracy: {round (results_epoch['Train Accuracy'], DECIMALS)}.")
    print (f"\tTrain Time: {int (results_epoch['Time (Train)'] / 60)} min & {int (results_epoch['Time (Train)'] % 60)} sec.")
    print (f"\tTest Loss: {round (results_epoch['Test Loss'], DECIMALS)}.")
    print (f"\tTest Accuracy: {round (results_epoch['Test Accuracy'], DECIMALS)}.")
    print (f"\tTest Time: {int (results_epoch['Time (Test)'] / 60)} min & {int (results_epoch['Time (Test)'] % 60)} sec.")
    print (f"------ ENDING EPOCH {results_epoch['Epoch']} ------")
    print ("\n\n")

# --------------- 
# MAIN CODE
# ---------------

def main_code ():
    # Variables
    path = r"C:\Users\Usuario\Desktop\proyectos\Brain_Tumor_Classification\data"    # datset path
    
    # Preparate the model
    neuralNet = ResNet50Classifier (dropout_prob=0.5, num_classes=len (labels))
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(neuralNet.parameters(), lr=0.001, weight_decay=1e-4)
    earlyStop = EarlyStop (patience=5, threshold_loss=0.001, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    
    # Create dataset instances with transforms
    train_dataset = BrainTumorDataset(os.path.join(path, 'Training'), transform=train_transform)
    test_dataset = BrainTumorDataset(os.path.join(path, 'Testing'), transform=test_transform)

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
        
        scheduler.step (test_loss)
        
        print_results (results)
        
        if earlyStop.checking_conditions (results.iloc[-1], neuralNet):
            break

    # Saving data
    save_results_into_disk (results)
    save(neuralNet.state_dict(), os.path.join(os.getcwd (), "models", "model.pth"))

# ------------------------
# EXECUTE MAIN CODE
# ------------------------

if __name__ == '__main__':
    main_code ()


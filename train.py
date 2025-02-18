import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from torch import save

from PIL import Image
import os

# Neural net modifiers
epochs = 10
batch_size = 32
learning_rate = 0.001
image_size = (224, 224)

# Constants
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']                       # List classes



# Class that manage the neural net
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet18Classifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Load pre-trained ResNet18
        in_features = self.model.fc.in_features  # Get the number of input features of the last layer
        self.model.fc = nn.Linear(in_features, num_classes)  # Replace the fc layer

    def forward(self, x):
        return self.model(x)

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
        image = Image.open (image_path)
        
        if self.transform:
            image = self.transform (image)
            
        return image, label 

def train (model, optimizer):
    model.train ()
    return









if __name__ == '__main__':
    # Variables
    path = r"C:\Users\Usuario\Desktop\proyectos\Brain_Tumor_Classification\data"    # datset path
    
    # Preparate the model
    neuralNet = ResNet18Classifier (len (labels))
    
    # Process to train the neural net
    for epoch in range (epochs):
       a = 0 
    
    save(neuralNet.state_dict(), os.path.join(os.getcwd (), "model1.pth"))



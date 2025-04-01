import torch.nn as nn
import torchvision.models as models
import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


# ----------------------------------
# NEURAL NET CLASS
# ----------------------------------

class ResNet50Classifier(nn.Module):
    def __init__(self, state_dict_path=None, num_classes=4):
        super(ResNet50Classifier, self).__init__()
        
        # Load pre-trained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(DEVICE)
        
        # Modify the fully connected layer to adjust for the number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes).to(DEVICE)
        
        # Load state_dict if provided
        if state_dict_path:
            self.load_state(state_dict_path)

    def forward(self, x):
        x = x.to(DEVICE)
        return self.model(x)
    
    def load_state(self, state_dict_path):
        """Loads the model's state dictionary from a given file path."""
        self.model.load_state_dict(torch.load(state_dict_path, map_location=DEVICE))
        print(f"Model state loaded from {state_dict_path}")




def predict ():

    model = ResNet50Classifier (state_dict_path=None, num_classes=4)
    print ("Fin")

if __name__ == '__main__':
    predict ()
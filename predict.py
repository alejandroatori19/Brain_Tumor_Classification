import torch.nn as nn
import torchvision.models as models
import torch

# Class that manage the neural net
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet18Classifier, self).__init__()
        self.model = models.resnet18()  # Load pre-trained ResNet18
        in_features = self.model.fc.in_features  # Get the number of input features of the last layer
        self.model.fc = nn.Linear(in_features, num_classes)  # Replace the fc layer

    def forward(self, x):
        return self.model(x)

model = ResNet18Classifier (4)
model.load_state_dict(torch.load("model1.pth"))  # Load the weights and biases from the saved state_dict
print ("Fin")
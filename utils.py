import os
from PIL import Image
from torch.utils.data import Dataset

# Here are saved the labels that image could have.
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Generate a custom dataset
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
                
        return
    
    # Return the length of the dataset
    def __len__(self):
        return len (self.data)
    
    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open (image_path)
        
        if self.transform:
            image = self.transform (image)
            
        return image, label 
    
    
# Method that helps wrapping the class
def load_dataset (path):
    return 
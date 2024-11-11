# %% [markdown]
# # Notebook for assignment Responsible AI predictive XAI exercise

# %%
import os
import matplotlib.pyplot as plt
import torch
import torchvision
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random

from utils.notebook import display_scrollable_dataframe, plot_sailency
from data_loaders import CUB_extnded_dataset
from models import get_inception_transform
from IPython.display import display

from sailency import get_saliency_maps, saliency_score_part

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# %%
# Settings for the experiment your running
data_set = 'val' 

Image_id = 406 # The id of the image you want to explain

# Settings for loading the dataset, make sure its the same as the one used to train the model. 
# You can find the settings in the .hydra folders config.yaml 
data_config = {'CUB_dir':r'data/CUB_200_2011',
                'split_file':r'data/train_test_val.pkl',
                'use_majority_voting':True,
                'min_class_count':10,
                'return_visibility':True}

# %%
#Define data set, the human transformer data_set is used to get the original images instead of the normalized ones
transformer = get_inception_transform(mode='val',methode="center")
human_tansform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(299),torchvision.transforms.ToTensor()])

#Get dataset
data = CUB_extnded_dataset(mode=data_set,config_dict=data_config,transform=transformer)

#Get a dataset that return original images instead normalized ones
data_human = CUB_extnded_dataset(mode=data_set,config_dict=data_config,transform=human_tansform)
concept_names = data.consept_labels_names
class_names = data.class_labels_names
n_classes = data.n_classes
n_concepts = data.n_concepts

model_folder = r"models/Sequential_Basemodel3"

X_to_C_path = os.path.join(model_folder,"best_XtoC_model.pth")
C_to_Y_path = os.path.join(model_folder,"best_CtoY_model.pth")

ModelXtoC = torch.load(X_to_C_path,map_location=torch.device('cpu'))
ModelCtoY = torch.load(C_to_Y_path,map_location=torch.device('cpu'))

# %%
# Import the necessary module
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

#Set the model to evaluation mode
ModelXtoC.eval()

nodes_list = ['x', 'Conv2d_3b_1x1.relu', 'Mixed_6b.cat', 'Mixed_7b.cat_2', 'fc']
return_nodes = {node: node for node in nodes_list}

# Create the feature extractor
feature_extractor = create_feature_extractor(ModelXtoC.model, return_nodes=return_nodes)

# Now, process the input image X through the feature extractor

# %% [markdown]
# ## Collect Features and Concepts from the Entire Dataset

# %%
from torch.utils.data import DataLoader

# Create a DataLoader for the dataset
batch_size = 1  # Adjust batch size as per your memory constraints

#Make data loader with fixed random seed for reproducibility
data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

# Initialize dictionaries to store features and lists for concepts
all_features = {name: [] for name in return_nodes.keys()}
all_classes = []
all_concepts = []

# Ensure the model is in evaluation mode
ModelXtoC.eval()
feature_extractor.eval()

# Use only 1% of the dataset for debugging
total_samples = len(data_loader)
debug_samples = max(1, total_samples)

# Iterate over the DataLoader
for i, (images, concepts, classes, _) in enumerate(tqdm(data_loader, desc="Collecting features and concepts")):
    if i >= debug_samples:
        break
    # Move images to device if using GPU
    images = images.to('cpu')  # Change 'cpu' to 'cuda' if using GPU

    # Get features from the feature extractor
    with torch.no_grad():
        outputs = feature_extractor(images)

    # Collect features from each layer
    for name in return_nodes.keys():
        layer_output = outputs[name]
        # Flatten the layer output
        layer_output_flat = layer_output.view(layer_output.size(0), -1)
        all_features[name].append(layer_output_flat.cpu())

    # Collect concepts
    all_classes.append(classes.cpu())
    all_concepts.append(concepts.cpu())

# Concatenate features and concepts
for name in return_nodes.keys():
    all_features[name] = torch.cat(all_features[name], dim=0)
    print(f"Features from layer {name} collected. shape: {all_features[name].shape}")
all_classes = torch.cat(all_classes, dim=0)
all_concepts = torch.cat(all_concepts, dim=0)

print("Features and concepts collected from the dataset.")

# %% [markdown]
# ## Train Linear Models to Predict Concepts from Intermediate Features

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Convert concepts to numpy array
# Convert one-hot encoded classes to class labels
# Separate the concepts and their visibility only relevant if visibility is used and no majority voting is used
if len(all_concepts.shape) == 3:
    all_concepts = all_concepts[:,0,:]

classes_np = all_classes.numpy()
concepts_np = all_concepts.numpy()

# Initialize a dictionary to store errors
layer_class_errors = {name: [] for name in return_nodes.keys()}
layer_concept_errors = {name: [] for name in return_nodes.keys()}

# Iterate over the layers
from tqdm import tqdm
import torch.nn.functional as F

for name in tqdm(return_nodes.keys(), desc="Layers"):
    print(f"\nTraining on features from layer: {name}")
    # Get features from this layer
    features_tensor = all_features[name]
    features_np = features_tensor.numpy()

    # Due to high dimensionality, perform dimensionality reduction
    from sklearn.decomposition import PCA

    # Adjust n_components based on the feature size
    n_components = min(100, features_np.shape[1], features_np.shape[0])
    pca = PCA(n_components=n_components)
    features_reduced = pca.fit_transform(features_np)

    # Split into train and test sets
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train_classes, y_test_classes = train_test_split(
        features_reduced, classes_np, test_size=0.2, random_state=42
    )

    #First train logistic regression for classes
    # Train logistic regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, np.argmax(y_train_classes, axis=1))

    # Predict probabilities and evaluate error using cross-entropy loss
    y_pred_prob = clf.predict_proba(X_test)
    y_test_tensor = torch.tensor(y_test_classes)
    #Extend predic prob to all 200 classes
    y_pred_prob_tensor = torch.zeros((y_test_classes.shape))

    for idx, class_idx in enumerate(clf.classes_):
        y_pred_prob_tensor[:,class_idx] = torch.tensor(y_pred_prob[:,idx])
    epsilon = 1e-12
    y_pred_prob_tensor = torch.clamp(y_pred_prob_tensor, epsilon, 1. - epsilon)
    
    loss = -torch.sum(y_test_tensor * torch.log(y_pred_prob_tensor), dim=1).mean().item()

    layer_class_errors[name].append(loss)
    print(
        f"layer {name}: Cross-Entropy Loss classes: {loss:.4f}"
    )

# %% [markdown]
# ## Analyze the Results

# %%
import pandas as pd

# Create a DataFrame to display the errors
results = pd.DataFrame(layer_class_errors)
print("\nCross-Entropy Loss of predicting classes from different layers:")
display(results.T)

# Plot the errors for visualization
results.T.plot(kind='bar', figsize=(12, 6))
plt.ylabel('Cross-Entropy Loss')
plt.title('Class Prediction Cross-Entropy Loss from Different Layers')
plt.legend(title='Layers', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

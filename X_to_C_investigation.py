# %% [markdown]
# # Notebook for assignment Responsible AI predictive XAI exercise

# %%
import os
import matplotlib.pyplot as plt
import torch
import torchvision
import pandas as pd
import numpy as np

from utils.notebook import display_scrollable_dataframe,plot_sailency
from data_loaders import CUB_extnded_dataset
from models import get_inception_transform
from IPython.display import display

from sailency import get_saliency_maps,saliency_score_part

# %%
# Settings for the experiment your running
data_set = 'ckpt' 

Image_id = 406 # The id of the image you want to explain

# Settings for loading the dataset, make sure its the same as the one used to train the model. 
# You can find the settings in the .hydra folders config.yaml 
data_config = {'CUB_dir':r'data/CUB_200_2011',
                'split_file':r'data/train_test_val.pkl',
                'use_majority_voting':False,
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

# %%
X, C, Y,coordinates = data.get_by_image_id(Image_id)

print("Image shape: ",X.shape)
print("Concept shape: ",C.shape)
print("Label shape: ",Y.shape)
X = X.unsqueeze(0)

img ,_,_,_ = data_human.get_by_image_id(Image_id)

print(Y.argmax())

# %%
# Separate the concepts and their visibility only relevant if visibility is used and no majority voting is used
if len(C.shape) == 2:
    Concepts = C[0]
    Concepts_visiblity = C[1]
else:
    Concepts = C.tolist()
    Concepts_visiblity = [None]*len(C)


# %% [markdown]
# # Expandability for Sequential and independent models.  

# %%
#Load models make sure the model is trained with the same settings as the data loader
model_folder = r"models/Sequential_Basemodel2"

X_to_C_path = os.path.join(model_folder,"best_XtoC_model.pth")
C_to_Y_path = os.path.join(model_folder,"best_CtoY_model.pth")

ModelXtoC = torch.load(X_to_C_path,map_location=torch.device('cpu'))
ModelCtoY = torch.load(C_to_Y_path,map_location=torch.device('cpu'))


# %%
# Import the necessary module
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

# Set the model to evaluation mode
ModelXtoC.eval()

# Get all node names for both training and evaluation modes
train_nodes, eval_nodes = get_graph_node_names(ModelXtoC.model)

# Decide which mode you are in
current_mode = 'train' if ModelXtoC.training else 'eval'

# Select the appropriate node names
if current_mode == 'train':
    available_nodes = train_nodes
else:
    available_nodes = eval_nodes

print(f"Model is in {current_mode} mode.")
print("Available node names in the model:")
print(available_nodes)

# Create a return_nodes dictionary mapping each available node name to a unique key
# Exclude 'AuxLogits' if not present
return_nodes = {name: name for name in available_nodes}

# Create the feature extractor
feature_extractor = create_feature_extractor(ModelXtoC.model, return_nodes=return_nodes)

# Now, process the input image X through the feature extractor
features = feature_extractor(X)

# features is a dictionary with keys as layer names and values as the outputs from those layers
# Let's print out the keys to see the layers
print("Extracted features from layers:")
for layer_name in features.keys():
    print(layer_name)

# %% [markdown]
# ## Collect Features and Concepts from the Entire Dataset

# %%
# %%
from torch.utils.data import DataLoader

# Create a DataLoader for the dataset
batch_size = 16  # Adjust batch size as per your memory constraints
data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

# Initialize dictionaries to store features and lists for concepts
all_features = {name: [] for name in return_nodes.keys()}
all_concepts = []

# Ensure the model is in evaluation mode
ModelXtoC.eval()
feature_extractor.eval()

# Iterate over the DataLoader
for images, concepts, _, _ in data_loader:
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
    all_concepts.append(concepts.cpu())

# Concatenate features and concepts
for name in return_nodes.keys():
    all_features[name] = torch.cat(all_features[name], dim=0)
all_concepts = torch.cat(all_concepts, dim=0)

print("Features and concepts collected from the dataset.")


# %% [markdown]
# ## Train Linear Models to Predict Concepts from Intermediate Features

# %%
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Convert concepts to numpy array
concepts_np = all_concepts.numpy()

# For demonstration, let's predict a few concepts (e.g., first 5 concepts)
concept_indices = [0, 1, 2, 3, 4]  # Adjust as needed

# Initialize a dictionary to store accuracies
layer_accuracies = {name: [] for name in return_nodes.keys()}

# Iterate over the layers
for name in return_nodes.keys():
    print(f"\nTraining on features from layer: {name}")
    # Get features from this layer
    features_tensor = all_features[name]
    features_np = features_tensor.numpy()

    # Due to high dimensionality, perform dimensionality reduction
    from sklearn.decomposition import PCA

    # Adjust n_components based on the feature size
    n_components = min(100, features_np.shape[1])
    pca = PCA(n_components=n_components)
    features_reduced = pca.fit_transform(features_np)

    # Iterate over selected concepts
    for concept_idx in concept_indices:
        concept_labels = concepts_np[:, concept_idx]

        # Split into train and test sets
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            features_reduced, concept_labels, test_size=0.2, random_state=42
        )

        # Train logistic regression
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        layer_accuracies[name].append(accuracy)
        print(
            f"Concept {concept_names[concept_idx]} (Index {concept_idx}): Accuracy {accuracy:.4f}"
        )


# %% [markdown]
# ## Analyze the Results

# %%
# %%
import pandas as pd

# Create a DataFrame to display the accuracies
results = pd.DataFrame(layer_accuracies, index=[concept_names[i] for i in concept_indices])
print("\nAccuracy of predicting concepts from different layers:")
display(results.T)

# Plot the accuracies for visualization
results.T.plot(kind='bar', figsize=(12, 6))
plt.ylabel('Accuracy')
plt.title('Concept Prediction Accuracy from Different Layers')
plt.legend(title='Concepts', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


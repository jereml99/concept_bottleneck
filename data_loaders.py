"""


"""
import os
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader

from collections import defaultdict as ddict

class CUB_dataset(Dataset):
    """
    A basic CUB dataset for concept learning
    Will return the image, concepts and class labels 
    if visibility is true it will return a tuple of concepts and visibility
    """


    def __init__(self, mode, config_dict: dict,transform=None): 
        """
        mode: str,
        config_dict: dict, dictionary containing all the necessary information for the dataset
        transform: torchvision.transforms, transform to be applied to the image
        """
        self.n_classes = 200 # Number of classes in the dataset
        self.n_concepts = 312
        self.transform = transform

        self.image_dir = os.path.join(config_dict['CUB_dir'],'images') # 

        split = pickle.load(open(os.path.join(config_dict['split_file']), 'rb')) #Load the train test val split

        #Load id of images based on the train mode
        if mode == 'train':
            self.data_id = split['train']
        elif mode == 'test':
            self.data_id = split['test']
        elif mode == 'val':
            self.data_id = split['val']
        elif mode == 'ckpt':
            self.data_id = split['train'] + split['val']
        else:
            raise ValueError('mode must be either train, test, val or ckpt')
        
        # Perform majority voting if the config_dict specifies
        if config_dict['use_majority_voting']:
            self.majority_voting = True

            train_ids = split['train'] # Majorly voting is based on the training set

            concepts, visibility = self.load_concepts(config_dict['CUB_dir']) # Load the concepts and visibility labels
            train_labels = self.load_labels(config_dict['CUB_dir']) # Load the class labels

            self.concepts, self.concept_mask = self.apply_filter(train_ids,config_dict["min_class_count"],concepts,train_labels,visibility) # Apply filter to the class attributes
            self.visibility = None # Visibility is not relevant after majority voting

            self.n_concepts = len(self.concept_mask) # Update the number of concepts based on the filter
        else:
            self.majority_voting = False

            if config_dict['return_visibility']:
                self.concepts, self.visibility = self.load_concepts(config_dict['CUB_dir'])
            else:
                self.concepts, _ = self.load_concepts(config_dict['CUB_dir'])
            
                self.visibility = None

        
        self.labels = self.load_labels(config_dict['CUB_dir']) # Load the class labels
        self.image_paths = self.load_images_paths(config_dict['CUB_dir']) # Load the image paths

            

    def load_labels(self,data_dir: str):

        labels = ddict(list)


        with open(os.path.join(data_dir,'image_class_labels.txt'), 'r') as f:
            for line in f:
                file_id, class_label = line.strip().split()

                labels[int(file_id)] = int(class_label)-1 # -1 to make the labels 0 indexed

        return labels        

    def load_images_paths(self,data_dir: str):
            
            image_paths = ddict(str)

            with open(os.path.join(data_dir,'images.txt'), 'r') as f:
                for line in f:
                    file_id, image_path = line.strip().split()
                    image_paths[int(file_id)] = os.path.join(self.image_dir,image_path)

            return image_paths    


    def load_concepts(self,data_dir: str):
        """
        Function to load the concepts and visibility labels
        arguments:
        data_dir: path to the data directory
        ids: list of ids of the training set


        returns:
        concepts: dictionary of list of concepts as binary labels
        visibility: dictionary of list of visibility labels ranging from 0 to 1
        """
        
        # 1 = not visible, 2 = guessing, 3 = probably, 4 = definitely
        #uncertainty_map = {1: {1: 0, 2: 0.5, 3: 0.75, 4:1}, #calibrate main label based on uncertainty label
                    #0: {1: 0, 2: 0.5, 3: 0.25, 4: 0}}

        

        concepts = ddict(list)
        visibility = ddict(list)

        with open(os.path.join(data_dir,'attributes','image_attribute_labels.txt'), 'r') as f:
            for line in f:
                file_id, attribute_idx, attribute_label, attribute_certainty = line.strip().split()[:4]

                concepts[int(file_id)].append(int(attribute_label)) 

                visibility[int(file_id)].append(int(attribute_certainty))
        

        return concepts, visibility


    def apply_filter(self,ids:list, min_class_count:int,concept:dict,labels:dict,visibility:dict):
        """
        Function to apply filter to the class attributes based on the majority voting to match the original code
        arguments:
        ids: list of ids of the training set
        min_class_count: minimum number of classes that should have the attribute
        concept: dictionary of concepts
        labels: dictionary of labels
        visibility: dictionary of visibility

        returns:
        class_max_label: filtered a numpy array as a matrix where each row corresponds to a class and each column to a concept
        mask: mask of the filtered concepts
        """
        
        #Calculate matrix of class-concept counts
        class_concept_count = np.zeros((200, 312, 2)) # 200 classes, 312 concepts, 2 types of labels, assuming normal CUB dataset
        for i in ids:
            for concept_idx, concept_label in enumerate(concept[i]):
                #ignore concept if not visible
                if  visibility[i][concept_idx] == 1: 
                    continue
                else:
                    class_concept_count[labels[i],concept_idx,concept_label] += 1
        
        #Perfomes majority voting 
        class_max_label = np.argmax(class_concept_count, axis=2)

        #settles ties by setting the class attribute label to 1 if the count is equal
        class_min_label = np.argmin(class_concept_count, axis=2)
        equal_count = np.where(class_min_label == class_max_label)
        class_max_label[equal_count] = 1

        #Apply filter
        if min_class_count:
            class_count = np.sum(class_max_label, axis=0)
            mask = np.where(class_count >= min_class_count)[0] #select attributes that are present (on a class level) in at least [min_class_count] classes
        else:
            mask = np.arange(312)
        
        return class_max_label[:,mask], mask

    def calculate_imbalance(self):
        """
        Calculate class imbalance ratio for all binary concept labels.
        
        :return: A list of imbalance ratios, one for each concept.
        """
        n_samples = len(self.data_id)
        
        # Initialize a list to count the number of positive labels for each concept
        positive_count = [0] * self.n_concepts
        
        if self.majority_voting:
            # If majority voting is applied, the number of samples is equal to the number of classes
            n_samples = self.n_concepts

            # Count positive labels for each concept across all classes
            for y in range(self.n_classes):
                for i, label in enumerate(self.concepts[y]):
                    positive_count[i] += label

        
        else:
            # If majority voting is not applied, the number of samples is equal to the number of images
            n_samples = len(self.data_id)
            # Count positive labels for each concept across all samples
            for img_id in self.data_id:
                for i, label in enumerate(self.concepts[img_id]):
                    positive_count[i] += label

        # Calculate imbalance ratio for each concept
        imbalance_ratios = []
        for count in positive_count:
            if count > 0:
                # Imbalance ratio = (total samples / positive samples) - 1
                ratio = (n_samples / count) - 1
            else:
                # If no positive samples, set ratio to infinity
                ratio = float('inf')
            imbalance_ratios.append(ratio)

        return imbalance_ratios

    
    def __len__(self):
        return len(self.data_id)

    def __getitem__(self, idx):
        img_id = self.data_id[idx]


        img_path = self.image_paths[img_id]
        X = Image.open(img_path).convert('RGB')

        if self.transform:
            X = self.transform(X)
        else:
            X = transforms.ToTensor()(X)
        
        Y = self.labels[img_id]

        if self.majority_voting:
            C = self.concepts[Y] # If majority voting is applied the concepts are based on the class label
        else:
            #Make C a tuple if visibility is not None
            if self.visibility is not None:
                C = (self.concepts[img_id], self.visibility[img_id])
            else:
                C = self.concepts[img_id]
        
        #Make C a tensor
        C = torch.tensor(C, dtype=torch.float32)
        
        #Make Y one hot encoded
        Y_one_hot = torch.zeros(self.n_classes, dtype=torch.float32)
        Y_one_hot[Y] = 1



        return X, C, Y_one_hot

    
    


class CUB_CtoY_dataset(CUB_dataset):
    """
    Dataset class for CUB dataset for the end classifier C to Y thus it only returns the class labels and the concepts
    It  can take in a pre-trained X to C model to generate the concepts
    """

    def __init__(self, mode:str, config_dict: dict,transform=None,  model:str = None,device:str = 'cpu'): 
        """
        return_mode: str, 
        """
        super().__init__(mode,config_dict)


    def generate_concept_mask(self,model,device,hard_concept:bool = False):
        """
        Function to generate the concepts given an x to c model
        """
        new_concepts = []

        model.eval()

        #Iterate over the dataset
        for idx in range(len(self)):
            X, _, _ = super().__getitem__(idx) # Get the image from parent class
            X = X.unsqueeze(0)
            X = X.to(device)
            with torch.no_grad():
                output = model(X)
                
                #Round the output if hard_concept is True
                if hard_concept:
                    output = torch.round(output)

            new_concepts.append(output.cpu().numpy())

        return new_concepts

    def __getitem__(self, idx):
        img_id = self.data_id[idx]
        Y = self.labels[img_id]

        if self.majority_voting:
            C = self.concepts[Y] # If majority voting is applied the concepts are based on the class label
        else:
            #Make C a tuple if visibility is not None
            if self.visibility is not None:
                C = (self.concepts[img_id], self.visibility[img_id])
            else:
                C = self.concepts[img_id]

        #Make C a tensor
        C = torch.tensor(C, dtype=torch.float32)
        
        #Make Y one hot encoded
        Y_one_hot = torch.zeros(self.n_classes, dtype=torch.float32)
        Y_one_hot[Y] = 1

        return C, Y_one_hot 


class CUB_extnded_dataset(CUB_dataset):
    """
    A cup dataset that would return coordinates on concepts
    """

    def __init__(self,mode:str, config_dict: dict,transform=None,crop_size:int =299):
        """
        config_dict: dict, dictionary containing all the necessary information for the dataset
        transform: torchvision.transforms, transform to be applied to the image
        """

        self.crop_size = crop_size
        super().__init__(mode,config_dict,transform)

                #Read the file with the names of bird location attributes
        self.part_names = []
        with open(os.path.join(config_dict['CUB_dir'],"parts","parts.txt")) as f:
            for line in f:
                self.part_names.append(line.strip().split(" ")[1:]) #remove the first element which is the index of the attribute


        self.part_locations=pd.read_csv(os.path.join(config_dict['CUB_dir'],"parts","part_locs.txt"), sep=" ", header=None)
        self.part_locations.columns = ["id","part","x","y","visible"]

        self.part_names_single = ['back', 'beak', 'belly', 'breast', 'crown', 'forehead', 'eye', 'leg', 'wing', 'nape', 'tail', 'throat'] #List of all path without left and right

        #Load concept labels names
        self.consept_labels_names = pd.read_csv(os.path.join(config_dict['CUB_dir'],"attributes.txt"), sep=" ", header=None)[1].values

        #Load class labels names
        self.class_labels_names = pd.read_csv(os.path.join(config_dict['CUB_dir'],"classes.txt"), sep=" ", header=None)[1].values

        #If a filter was applied load the filter names.
        if self.majority_voting:
            self.consept_labels_names= self.consept_labels_names[self.concept_mask]


    def get_cordinates(self,id,img):
        """
        Get coordinates of the concepts

        Note this method assume the image is center cropped
        """

        coordinate_dict = {}

        for name in self.part_names_single:
            coordinate_dict[name] = {"coordinate":[],"visible":0}

        # Calculate crop coordinates
        orig_width, orig_height = img.size
        left = (orig_width - self.crop_size) // 2
        top = (orig_height - self.crop_size) // 2


        for index,part in self.part_locations[self.part_locations.id==id].iterrows(): #Get all the parts for the image
            name = self.part_names[int(part["part"])-1]
            
            x = part["x"] - left
            y = part["y"] - top

            if name[0] in self.part_names_single:


                if x >= 0 and x < self.crop_size and y >= 0 and y < self.crop_size and  part["visible"] == 1: #Check if the part is in the crop and visible
                    coordinate_dict[name[0]]["coordinate"].append((x,y))
                    coordinate_dict[name[0]]["visible"] = 1
                else:
                    coordinate_dict[name[0]]["visible"] = 0

            elif name[0] in ["left","right"]:
                if x >= 0 and x < self.crop_size and y >= 0 and y < self.crop_size and part["visible"] == 1: #Check if the part is in the crop
                    coordinate_dict[name[1]]["coordinate"].append((x,y))
                    coordinate_dict[name[1]]["visible"] = 1
                else:
                    coordinate_dict[name[1]]["visible"] = 0
            
        concept_coordinate = []

        for concept_name in self.consept_labels_names:
            concept_name = concept_name.split("_")[1] #Remove get part related to the concept

            #Apparently english has two words for the mouth of a bird and the dataset uses both. 
            if concept_name == "bill":
                concept_name = "beak"

            #Check if the concept is a part
            if concept_name in self.part_names_single or concept_name in ["eye","leg","wing"]:
                concept_coordinate.append(coordinate_dict[concept_name]["coordinate"])
            else:
                concept_coordinate.append([])
        return concept_coordinate
    
    def get_by_image_id(self,id):
        """
        Get the image by the image id from data/CUB_200_2011/images.txt
        """
        try:
            idx = self.data_id.index(id)
            return self.__getitem__(idx)
        except ValueError:
            raise ValueError("Image id not found in this dataset")
    
    def __getitem__(self, idx):

        # Get the image path from the image dataset
        # Removed code for modifying the image path not sure if it was important or just asumed dataset to be sorted
        img_id = self.data_id[idx]


        img_path = self.image_paths[img_id]
        img = Image.open(img_path).convert('RGB')

        
        Y = self.labels[img_id]

        if self.majority_voting:
            C = self.concepts[Y] # If majority voting is applied the concepts are based on the class label


        else:
            #Make C a tuple if visibility is not None
            if self.visibility is not None:
                C = (self.concepts[img_id], self.visibility[img_id])
            else:
                C = self.concepts[img_id]
        
        #Get the coordinates before applying the transform
        coordinates = self.get_cordinates(self.data_id[idx],img)

        if self.transform:
            X = self.transform(img)

        #Make C a tensor
        C = torch.tensor(C, dtype=torch.float32)
        
        #Make Y one hot encoded
        Y_one_hot = torch.zeros(self.n_classes, dtype=torch.float32)
        Y_one_hot[Y] = 1

        return X, C, Y_one_hot, coordinates

    

# Test the code
if __name__ == "__main__":
    #Make sure majority voting is working is returning the exact dataset as the original papar

    #Load original pickle file
    import pickle

    original = pickle.load(open(r'data\CUB_processed\Original\train.pkl','rb'))

    config_dict = {'CUB_dir':r'data/CUB_200_2011','split_file':r'data\train_test_val.pkl','use_majority_voting':True,'min_class_count':10,'return_visibility':True}
    transform = transforms.Compose([transforms.Resize((299,299)),transforms.ToTensor()])
    dataset = CUB_dataset('train',config_dict,transform)
    
    #assert the length of the dataset is the same as the original
    assert len(dataset) == len(original) 

    #assert the first same amount of concepts
    assert len(dataset.__getitem__(0)[1]) == len(original[0000]["attribute_label"]) , f"{len(dataset.__getitem__(0)[1])} != {len(original[0000]['attribute_label'])}"

    # Check if the concepts are the same for the first image
    assert [np.int64(i) for i in dataset.__getitem__(0)[1]] == original[0000]["attribute_label"], f"{dataset.__getitem__(0)} != {original[0000]['attribute_label']}"

    #Check if the class label is the same
    assert dataset.__getitem__(0)[2].argmax().item() == original[0000]["class_label"], f"{dataset.__getitem__(0)[2]} != {original[0000]['class_label']}"

    #Check the validation set
    dataset = CUB_extnded_dataset('val',config_dict,transform)
    original = pickle.load(open(r'data\CUB_processed\Original\val.pkl','rb'))

    #assert the length of the dataset is the same as the original
    assert len(dataset) == len(original) , f"{len(dataset)} != {len(original)}"

    #assert the first same amount of concepts
    assert len(dataset.__getitem__(0)[1]) == len(original[0000]["attribute_label"]) , f"{len(dataset.__getitem__(0)[1])} != {len(original[0000]['attribute_label'])}"

    # Check if the concepts are the same for the first image
    assert [np.int64(i) for i in dataset.__getitem__(0)[1]] == original[0000]["attribute_label"], f"{dataset.__getitem__(0)} != {original[0000]['attribute_label']}"

    #Check if the class label is the same
    assert dataset.__getitem__(0)[2].argmax().item() == original[0000]["class_label"], f"{dataset.__getitem__(0)[2]} != {original[0000]['class_label']}"


    
    #Check if it works witout majority voting
    config_dict = {'CUB_dir':r'data/CUB_200_2011','split_file':r'data\train_test_val.pkl','use_majority_voting':False,'min_class_count':10,'return_visibility':False}
    transform = transforms.Compose([transforms.Resize((299,299)),transforms.ToTensor()])
    dataset = CUB_dataset('val',config_dict,transform)

    #assert the length of the dataset is the same as the original
    assert len(dataset) == len(original)

    #Check if ther is 312 concepts
    assert len(dataset.__getitem__(0)[1]) == 312

    #Check if there is 200 classes
    assert len(dataset.__getitem__(0)[2]) == 200


    dataset = CUB_extnded_dataset('val',config_dict,transform)
    print(dataset.calculate_imbalance())

    dataset = CUB_CtoY_dataset('val',config_dict,transform)

    #assert the length of the dataset is the same as the original
    assert len(dataset) == len(original)

    #Check if ther is 312 concepts
    assert len(dataset.__getitem__(0)[0]) == 312

    #Check if there is 200 classes
    assert len(dataset.__getitem__(0)[1]) == 200


        
            




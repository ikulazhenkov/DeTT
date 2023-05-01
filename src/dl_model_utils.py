import numpy as np
import os
import PIL
from PIL import Image

import time, gc
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset



def get_transform_train(in_size=224):
    """Return transformations for cholec80 train dataset creation

    Returns
    -------
    list : list of transformations to be applied to dataset
    """
    return transforms.Compose([
    transforms.CenterCrop(in_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.4345991,0.28689542,0.28143004],[0.20264472,0.18753591,0.18246838])
])
 

def get_transform_valid(in_size=224):
    """Return transformations for cholec80 validation dataset creation

    Returns
    -------
    list : list of transformations to be applied to dataset
    """
    return transforms.Compose([
    transforms.CenterCrop(in_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.4381856,0.2710791,0.27274346],[0.21493198,0.20029972,0.19810152])
])

def get_transform_test(in_size=224):
    """Return transformations for cholec80 test dataset creation

    Returns
    -------
    list : list of transformations to be applied to dataset
    """
    return transforms.Compose([
    transforms.CenterCrop(in_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.390472 ,0.2648226,0.25951037],[0.17919387,0.16188985,0.15494719])
])


def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print():
    torch.cuda.synchronize()
    end_time = time.time()
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))

class Cholec80Dataset(Dataset):
    '''
    Dataset Definition 
    '''
    def __init__(self, df, transform=None):
        self.image_path = df['image']
        self.phase_annotations = df['phase']
        self.tool_annotations = df[[
        'tool_Grasper',
        'tool_Bipolar',
        'tool_Hook',
        'tool_Scissors',
        'tool_Clipper',
        'tool_Irrigator',
        'tool_SpecimenBag']]
        # self.tool_target = df['tool_target']
        self.transform = transform

    def __len__(self):
        return len(self.phase_annotations)


    def __getitem__(self, index):
        img_path = self.image_path[index]
        img = Image.open(img_path).convert("RGB")
        phase_label = self.phase_annotations[index]
        # tool_target = self.tool_target[index]
        tool_label = self.tool_annotations.iloc[index].values
        tool_label = tool_label.tolist()
        tool_label = torch.FloatTensor(tool_label)
        if self.transform is not None:
            img = self.transform(img)

        return (img, img_path, phase_label, tool_label)


# class Cholec80Dataset(Dataset):
#     '''
#     Dataset Definition 
#     '''
#     def __init__(self, df, transform=None):
#         self.image_path = df['image']
#         self.phase_annotations = df['phase']
#         self.tool_annotations = df[[
#         'tool_Grasper',
#         'tool_Bipolar',
#         'tool_Hook',
#         'tool_Scissors',
#         'tool_Clipper',
#         'tool_Irrigator',
#         'tool_SpecimenBag']]
#         self.tool_target = df['tool_target']
#         self.transform = transform

#     def __len__(self):
#         return len(self.phase_annotations)


#     def __getitem__(self, index):
#         img_path = self.image_path[index]
#         img = Image.open(img_path).convert("RGB")
#         phase_label = self.phase_annotations[index]
#         tool_target = self.tool_target[index]
#         tool_label = self.tool_annotations.iloc[index].values
#         tool_label = tool_label.tolist()
#         tool_label = torch.FloatTensor(tool_label)
#         if self.transform is not None:
#             img = self.transform(img)

#         return (img, img_path, phase_label, tool_label,tool_target)





class EarlyStopping():
    """
    Early stopping
    """
    def __init__(self, patience=3, delta=0):
        """
        :param patience: how many epochs to wait before stopping job
        :param delta: min difference between new and previous loss for
                update to occur. Defaults at 0
        """
        self.patience = patience
        self.loss_delta = delta
        self.patience_counter = 0
        self.min_loss = None
        self.stop = False

    def __call__(self, val_loss, model, model_name, hmm_df):
        #Set the loss value on first iteration
        if self.min_loss == None:
            self.min_loss = val_loss
        #Update the loss value if better loss discovered. Also save the model
        elif self.min_loss - val_loss > self.loss_delta:
            self.save_model(val_loss, model, model_name, hmm_df)
            self.min_loss = val_loss
            self.patience_counter = 0
        #If loss is worse raise the patience counter
        #if patience counter reached stop execution.
        else:
            self.patience_counter += 1
            print("Early Stopping count is at: {} maximum is: {}".format(self.patience_counter, self.patience))
            # if self.patience_counter >= self.patience:
            #     print('Stopping Run')
            #     self.stop = True

    #If min_loss decreases we save the model
    def save_model(self, val_loss, model, model_name, hmm_df):
        #Saves model when validation loss decrease.
        print("Validation loss improved from {:.2f} to {:.2f}  Saving Model".format(self.min_loss,val_loss ))
#             _save_model(model, args.model_dir)  
        path  = os.path.join("models", f"{model_name}_model.pth")
        hmm_path = os.path.join("hmm_folder", f'{model_name}_hmm_df.parquet')
        torch.save(model.state_dict(), path)
        hmm_df.to_parquet(hmm_path)  
        


import torch     
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset

# This file is going to handle the dataset for enigma.

# Our dataModule is going to have a dataset that is initialised by 3 rotors, and a reflector.

class EnigmaDataModule(pl.LightningDataModule):
    def __init__(self,rotors = None ,reflector=None):
        super().__init__()
        self.rotors=rotors
        self.reflector=reflector
        if rotors is None:
            rotorA=torch.randperm(26)
            rotorB=torch.randperm(26)
            rotorC=torch.randperm(26)
            reflector=torch.randperm(26)
            #The rule of the reflector is that no letter can map to itself.
            while (reflector==torch.arange(26)).sum()>0:
                reflector=torch.randperm(26)

            self.rotors=[rotorA,rotorB,rotorC]

        #These constitute the GROUND Truth Enigma settings. 
        self.dataset=EnigmaDataset(rotors,reflector)

    def train_dataloader(self):
        #This is the dataloader that will be used for training.

        # There are some other flags that may be worth playing with, such as num_workers, pin_memory, and prefetch_factor, what do they do?
        return torch.utils.data.DataLoader(self.dataset,batch_size=16)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset,batch_size=16)
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset,batch_size=16)
    

class EnigmaDataset(torch.utils.data.IterableDataset):
    def __init__(self,rotors,reflector):
        super().__init__()
        self.rotors=rotors
        self.reflector=reflector
        # make these into 2d arrays with one_hot encoding
        self.rotors=[torch.nn.functional.one_hot(rotor,26) for rotor in rotors]
        self.reflector=torch.nn.functional.one_hot(reflector,26)
        self.rotor_positions=[0,0,0]
        rotateMatrix=torch.arange(26)+1 #This is the matrix that will be used to rotate the rotors.
        rotateMatrix[-1]=0
        self.rotateMatrix=torch.nn.functional.one_hot(rotateMatrix,26)
    def __iter__(self):
        return self
    def __next__(self):
        #This is the logic for the enigma machine
        #We will generate a sequence of 50 random letters, then encode it.

        GROUND_TRUTH= torch.randint(0,26,(150,))
        encoded=GROUND_TRUTH.clone()
        for i in range(150):
            #Rotate the rotors
            self.rotor_positions[0]=(self.rotor_positions[0]+1)%26
            if self.rotor_positions[0]==0:
                self.rotor_positions[1]=(self.rotor_positions[1]+1)%26
                if self.rotor_positions[1]==0:
                    self.rotor_positions[2]=(self.rotor_positions[2]+1)%26
            #Encode the letter
            encoded[i]=self.rotors[2][self.rotor_positions[2]].dot(self.rotors[1][self.rotor_positions[1]].dot(self.rotors[0][self.rotor_positions[0]].dot(self.reflector.dot(self.rotors[0][self.rotor_positions[0]].dot(self.rotors[1][self.rotor_positions[1]].dot(self.rotors[2][self.rotor_positions[2]]))))))[GROUND_TRUTH[i]]
        ## To Do: 
        # -  Add noise to the encoded message? Try randomizing a character in the encoded message, to represent a mistake in the encoding.
        # -  Try doing the above step with linear algebra, to make it faster and remove the for loop and the if statements.
        
        
        
        
        return encoded,GROUND_TRUTH
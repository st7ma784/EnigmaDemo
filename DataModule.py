
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
            
            self.rotors=[rotorA,rotorB,rotorC]
        if reflector is None:
            reflector=torch.randperm(26)
            #The rule of the reflector is that no letter can map to itself.
            while (reflector==torch.arange(26)).sum()>0:
                reflector=torch.randperm(26)
            self.reflector=reflector
        #These constitute the GROUND Truth Enigma settings. 
        self.dataset=EnigmaDataset(self.rotors,self.reflector)

    def train_dataloader(self):
        #This is the dataloader that will be used for training.

        # There are some other flags that may be worth playing with, such as num_workers, pin_memory, and prefetch_factor, what do they do?
        return torch.utils.data.DataLoader(self.dataset,batch_size=16,num_workers=4,pin_memory=True)
    
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
        self.rotors=rotors
        self.reflector=reflector
        rotateMatrix=torch.arange(26)+1 #This is the matrix that will be used to rotate the rotors.
        rotateMatrix[-1]=0
        self.rotateMatrix=torch.nn.functional.one_hot(rotateMatrix,26)
    def __iter__(self):
        return self
    def __len__(self):
        return 10000
    def __next__(self):
        #This is the logic for the enigma machine
        #We will generate a sequence of 50 random letters, then encode it.

        GROUND_TRUTH= torch.randint(0,26,(150,),dtype=torch.long)
        encoded=GROUND_TRUTH.clone()
        rotor_positions=[0,0,0]
        for i in range(150):
            #Rotate the rotors
            rotor_positions[0]=(rotor_positions[0]+1)%26
            if rotor_positions[0]==0:
                rotor_positions[1]=(rotor_positions[1]+1)%26
                if rotor_positions[1]==0:
                    rotor_positions[2]=(rotor_positions[2]+1)%26
            #Encode the letter
            letterAfterFirstRotor= self.rotors[0][(rotor_positions[0]+encoded[i])%26]
            letterAfterSecondRotor= self.rotors[1][(rotor_positions[1] + letterAfterFirstRotor)%26]
            letterAfterThirdRotor= self.rotors[2][(rotor_positions[2] + letterAfterSecondRotor)%26]
            reflectedLetter=self.reflector[letterAfterThirdRotor]
            #Now we have to go back through the rotors
            letterReturningThroughThirdRotor=self.rotors[2][(rotor_positions[2]+reflectedLetter)%26]
            letterReturningThroughSecondRotor=self.rotors[1][(rotor_positions[1]+letterReturningThroughThirdRotor)%26]
            letterReturningThroughFirstRotor=self.rotors[0][(rotor_positions[0]+letterReturningThroughSecondRotor)%26]
            #This is the encoded letter
            encoded[i]=letterReturningThroughFirstRotor

        ## To Experiment with: 
        # -  Add noise to the encoded message? Try randomizing a character in the encoded message, to represent a mistake in the encoding.
        # -  Try doing the above step with linear algebra, to make it faster and remove the for loop and the if statements.
        # -        -- HINT: You may want to think of the rotors as 1-hot encoded matrices of shape 26,26 and the encoding as a matrix multiplication. 
              
        
        return encoded,GROUND_TRUTH
    
if __name__=="__main__":
    dm=EnigmaDataModule()
    dl=dm.train_dataloader()
    for i in dl:
        print(i)
        break
    print("done")
    #print(dl)
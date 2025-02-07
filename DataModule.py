
import torch     
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset

# This file is going to handle the dataset for enigma.
# Our dataModule is going to have a dataset that is initialised by 3 rotors, and a reflector.

class EnigmaDataModule(pl.LightningDataModule):
    def __init__(self,rotors = None ,reflector=None,batch_size=32):
        super().__init__()
        self.rotors=rotors
        self.reflector=reflector
        self.batch_size=batch_size
        if rotors is None:
            rotorA=torch.randperm(26)
            rotorB=torch.randperm(26)
            rotorC=torch.randperm(26)
            
            self.rotors=[rotorA,rotorB,rotorC]
        if reflector is None:
            self.reflector=torch.arange(26)
            while torch.any(self.reflector-torch.arange(26)==0):
                #swap 2 random elements on the diagonal of the reflector
                indexes_to_swap=torch.arange(26)-self.reflector ==0
                indexes_to_swap=indexes_to_swap.to(torch.long)
                #swap the elements
                #first location is the first true in the indexes to swap
                firstLocation=torch.argmax(indexes_to_swap)
                #second location is a random int in indexes to swap that is not the first location
                secondLocations=torch.randperm(26)
                index=0
                while (secondLocations[index]==firstLocation or not indexes_to_swap[secondLocations[index]]):
                    index+=1
                    #print(secondLocation)
                #swap the elements
                a=self.reflector[firstLocation].item()
                b=self.reflector[secondLocations[index]].item()
                self.reflector[secondLocations[index]]=a
                self.reflector[firstLocation]=b

            #check that the reflector has all numbers 1 to 26
            assert torch.sum(torch.pow(self.reflector,2))==torch.sum(torch.pow(torch.arange(26),2))
            
        #These constitute the GROUND Truth Enigma settings. 
        self.dataset=EnigmaDataset(self.rotors,self.reflector)

    def train_dataloader(self):
        #This is the dataloader that will be used for training.

        # There are some other flags that may be worth playing with, such as num_workers, pin_memory, and prefetch_factor, what do they do?
        return torch.utils.data.DataLoader(self.dataset,batch_size=self.batch_size,num_workers=12,pin_memory=True,prefetch_factor=4)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset,batch_size=self.batch_size)
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset,batch_size=self.batch_size)
    

class EnigmaDataset(torch.utils.data.IterableDataset):
    def __init__(self,rotors,reflector):
        super().__init__()
        self.rotors=rotors
        self.reflector=reflector
        # make these into 2d arrays with one_hot encoding
        self.rotors=rotors
        self.reflector=reflector
    def __iter__(self):
        return self
    def __len__(self):
        return 10000
    def __next__(self):
        #This is the logic for the enigma machine
        #We will generate a sequence of 50 random letters, then encode it.

        GROUND_TRUTH= torch.randint(0,26,(11*26,),dtype=torch.long)
        encoded=self.ENCODEGroudTruth(GROUND_TRUTH)

        ## To Experiment with: 
        # -  Add noise to the encoded message? Try randomizing a character in the encoded message, to represent a mistake in the encoding.
        # -  Try doing the above step with linear algebra, to make it faster and remove the for loop and the if statements.
        # -        -- HINT: You may want to think of the rotors as 1-hot encoded matrices of shape 26,26 and the encoding as a matrix multiplication. 
              
        
        return encoded,GROUND_TRUTH
    def ENCODEGroudTruth(self,GROUND_TRUTH):
        encoded=torch.clone(GROUND_TRUTH)
        rotor_positions=[0,0,0]
        for i in range(11*26):
            #Rotate the rotors
            rotor_positions[0]=(i)%26
            rotor_positions[1]=(i//7)%26
            rotor_positions[2]=(i//11)%26
            #Encode the letter
            letterAfterFirstRotor= self.rotors[0][(rotor_positions[0]+encoded[i])%26]
            letterAfterSecondRotor= self.rotors[1][(rotor_positions[1] + letterAfterFirstRotor)%26]
            letterAfterThirdRotor= self.rotors[2][(rotor_positions[2] + letterAfterSecondRotor)%26]
            reflectedLetter=self.reflector[letterAfterThirdRotor]
            #Now we have to go back through the rotors
            #meaning we find the index of the number, 
            
            letterReturningThroughThirdRotor=(self.rotors[2][rotor_positions[2]:].tolist() + self.rotors[2][:rotor_positions[2]].tolist()).index(reflectedLetter)
            letterReturningThroughSecondRotor=(self.rotors[1][rotor_positions[1]:].tolist() + self.rotors[1][:rotor_positions[1]].tolist()).index(letterReturningThroughThirdRotor)
            letterReturningThroughFirstRotor=(self.rotors[0][rotor_positions[0]:].tolist() + self.rotors[0][:rotor_positions[0]].tolist()).index(letterReturningThroughSecondRotor)
            #This is the encoded letter
            encoded[i]=letterReturningThroughFirstRotor
        return encoded


if __name__=="__main__":
    import matplotlib.pyplot as plt
    dm=EnigmaDataModule()
    dl=dm.train_dataloader()
    for i in dl:
        fig= plt.figure()
        encoded,ground_truth=i
        ax1=fig.add_subplot(121)
        ax2=fig.add_subplot(122)
        ax1.imshow(encoded)
        ax2.imshow(ground_truth)
        plt.show()
        break
    print("Checking Properties")
    GROUND_TRUTH= torch.randint(0,26,(11*26,),dtype=torch.long)
    encoded=dm.dataset.ENCODEGroudTruth(GROUND_TRUTH)

    assert torch.allclose(dm.dataset.ENCODEGroudTruth(encoded),GROUND_TRUTH)
    print("All tests passed")
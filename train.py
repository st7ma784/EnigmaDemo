

from pytorch_lightning import LightningModule
import torch.nn as nn
import torch

class Rotor(nn.Module):
    def __init__(self,position=1):
        super().__init__()
        self.rotor=nn.parameter.Parameter(torch.rand(26,26))
        self.rotor.requires_grad=True
        self.position=position
        if position==0: #This is the default position
            self.generate_rotation_matrix=self.generate_rotation_matrixPos0
        elif position==1:
            self.generate_rotation_matrix=self.generate_rotation_matrixPos1
        elif position==2:
            self.generate_rotation_matrix=self.generate_rotation_matrixPos2
        elif position==3:
            self.generate_rotation_matrix=self.generate_rotation_matrixPos3
        else:
            raise ValueError("Invalid rotor position")
        self.rotation_matrix=torch.arange(26)+1 
        
    def generate_rotation_matrixPos0(self,sequence_length):
        return torch.ones(sequence_length,dtype=torch.long)
    def generate_rotation_matrixPos1(self,sequence_length):
        return torch.arange(sequence_length,dtype=torch.long)%26
    def generate_rotation_matrixPos2(self,sequence_length):
        return (torch.arange(sequence_length,dtype=torch.long)//26) %26
    def generate_rotation_matrixPos3(self,sequence_length):
        return (torch.arange(sequence_length,dtype=torch.long)//676) %26
    def forward(self,signalInput): 
        #signalInput is a tensor of shape (B,S,26) 
        Batch_Size=signalInput.shape[0]
        Sequence_Length=signalInput.shape[1]
        #Step 1: generate the rotation matrix
        rotationMatrix=torch.arange(26,dtype=torch.long).unsqueeze(0) + self.generate_rotation_matrix(Sequence_Length).unsqueeze(1)
        #Step 2: ensure the rotation matrix is <26
        rotationMatrix=rotationMatrix%26
        rotationMatrix=torch.nn.functional.one_hot(rotationMatrix,26).float()
        # shape is now (sequence_length,26,26)
        rotatedRotor=rotationMatrix@self.rotor
        #shape is now (sequence_length,26,26) 
        #repeat the rotor for the batch size
        rotatedRotor=rotatedRotor.unsqueeze(0).repeat(Batch_Size,1,1,1).view(-1,26,26) #shape is now (Batch_Size*sequence_length,26,26)
        #shape is now (Batch_Size,sequence_length,26,26)
        #the input signal is shape (Batch_Size,Sequence_Length,26)
        signalInput=signalInput.view(-1,1,26)
        return torch.bmm(signalInput,rotatedRotor).view(Batch_Size,Sequence_Length,26)

    def reverseRotorForward(self,signalInput): 
        #signalInput is a tensor of shape (B,S,26) 
        Batch_Size=signalInput.shape[0]
        Sequence_Length=signalInput.shape[1]
        rotationMatrix=torch.arange(26).unsqueeze(0) + self.generate_rotation_matrix(Sequence_Length).unsqueeze(1)
        rotationMatrix=rotationMatrix%26
        rotationMatrix=torch.nn.functional.one_hot(rotationMatrix,26).float()
        rotatedRotor=rotationMatrix@self.rotor

        rotatedRotor=rotatedRotor.unsqueeze(0).repeat(Batch_Size,1,1,1).view(-1,26,26).permute(0,2,1) # same as the above, but by transposing the last two dimensions, we get the reverse rotor.
        signalInput=signalInput.view(-1,1,26)
        return torch.bmm(signalInput,rotatedRotor).view(Batch_Size,Sequence_Length,26)
    

class Enigma(LightningModule):

    def __init__(self,optimizer_name,learning_rate,batch_size,precision,activation,lossName="CrossEntropy"):
        super().__init__()
        self.optimizer_name=optimizer_name
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.precision=precision
        self.activation=activation
        self.lossName=lossName
        if lossName=="CrossEntropy":
            self.loss=nn.CrossEntropyLoss(reduction="mean")
        elif lossName=="MSELoss":
            self.loss=nn.MSELoss(reduction="mean")
        else:
            raise ValueError("Invalid loss function")

        self.R1=Rotor(1)
        self.R2=Rotor(2)
        self.R3=Rotor(3)
        self.REF=Rotor(0)



        if activation=="gelu":
            self.activation=nn.GELU()
        elif activation=="relu":
            self.activation=nn.ReLU()
        elif activation=="sigmoid":
            self.activation=nn.Sigmoid()
        else:
            raise ValueError("Invalid activation function")

    def configure_optimizers(self):
    
        if self.optimizer_name=="adam":
            return torch.optim.Adam(self.parameters(),lr=self.learning_rate)
        elif self.optimizer_name=="sgd":
            return torch.optim.SGD(self.parameters(),lr=self.learning_rate)
        else:
            raise ValueError("Invalid optimizer name")

    def forward(self,x):
       
        x=self.R1(x)
        x=self.R2(x)
        x=self.R3(x)
        x=self.REF(x)
        x=self.R3.reverseRotorForward(x)
        x=self.R2.reverseRotorForward(x)
        x=self.R1.reverseRotorForward(x)

        #To DO: Add the activation function here. 
        x=self.activation(x) 
        # Test whether this is necessary, and whether it should be applied at each step, or just at the end.
        #reshape the tensor back to the original shape
        return x
    
    def training_step(self,batch,batch_idx):
        encoded,GT=batch #GT is the ground truth     -though the elegance of the enigma machine is that these can be the other way around and it will still work. 
        encoded=torch.nn.functional.one_hot(encoded,26).float()
        decoded=self.forward(encoded)
        loss=self.loss(decoded.permute(0,2,1),GT)
        self.log("loss",loss,prog_bar=True)
        return loss
    
    def print_enigma_settings(self):
        print("Rotor 1 : ",self.rotor_1.max(1).indices)
        print("Rotor 2 : ",self.rotor_2.max(1).indices)
        print("Rotor 3 : ",self.rotor_3.max(1).indices)
        print("Reflector : ",self.reflector)
        print("Activation : ",self.activation)
        print("Optimizer : ",self.optimizer_name)
        print("Learning Rate : ",self.learning_rate)
        print("Batch Size : ",self.batch_size)
        print("Precision : ",self.precision)
        print("Activation : ",self.activation)

    def compare_to_gt(self,rotors,reflector):
        #This function will take a batch of ground truth rotors and reflectors, and compare them to the current settings.
        rotor1Loss=torch.nn.functional.cross_entropy(self.R1.rotor,rotors[0])
        rotor2Loss=torch.nn.functional.cross_entropy(self.R2.rotor,rotors[1])
        rotor3Loss=torch.nn.functional.cross_entropy(self.R3.rotor,rotors[2])
        reflectorLoss=torch.nn.functional.cross_entropy(self.REF.rotor,reflector)
        self.log("Rotor 1 Loss",rotor1Loss)
        self.log("Rotor 2 Loss",rotor2Loss)
        self.log("Rotor 3 Loss",rotor3Loss)
        self.log("Reflector Loss",reflectorLoss)

        return rotor1Loss,rotor2Loss,rotor3Loss,reflectorLoss

        #We will compare the current settings to the ground truth settings.




if __name__ == "__main__":
    #build arguments
    import argparse
    import pytorch_lightning as pl
    from DataModule import EnigmaDataModule
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer_name",type=str,default="adam")
    parser.add_argument("--learning_rate",type=float,default=1e-4)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--precision",type=str,default="32")
    parser.add_argument("--activation",type=str,default="gelu")
    parser.add_argument("--loss",type=str,default="CrossEntropy")
    args=parser.parse_args()
    model=Enigma(args.optimizer_name,args.learning_rate,args.batch_size,args.precision,args.activation,args.loss)
    model.print_enigma_settings()
    print("Model Created Successfully")
    print("Model : ",model)

    dm=EnigmaDataModule()
    dm.setup(stage="train")
    print("DataModule Created Successfully")
    print("DataModule : ",dm)
   
    #login as anonymous
    wandb.login(anonymous="allow")
    wandb_logger = WandbLogger(project='Enigma', entity='st7ma784')

    #enable Early stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='loss',
        patience=3,
        verbose=False,
        mode='min'
    )

    #enable model checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='loss',
        dirpath='./',
        filename='Enigma-{epoch:02d}-{loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    #And finally a nice loading bar
    progress_bar = pl.callbacks.TQDMProgressBar()


    trainer=pl.Trainer(max_epochs=10,
                       max_steps=3000,
                        devices="auto",
                        accelerator="auto",
                        logger=wandb_logger,
                        callbacks=[early_stop_callback,checkpoint_callback,progress_bar])
                        # have a look at the other flags that can be set here.

    
    trainer.fit(model,dm)

    model.compare_to_gt(dm.rotors,dm.reflector)

    print("Model Trained Successfully")


        

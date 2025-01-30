

from pytorch_lightning import LightningModule
import torch.nn as nn
import torch

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
            self.loss=nn.CrossEntropyLoss()
        elif lossName=="MSELoss":
            self.loss=nn.MSELoss()
        else:
            raise ValueError("Invalid loss function")
        self.rotor_1=nn.parameter.Parameter(torch.rand(26,26))
        self.rotor_2=nn.parameter.Parameter(torch.rand(26,26))
        self.rotor_3=nn.parameter.Parameter(torch.rand(26,26))
        self.reflector=nn.parameter.Parameter(torch.rand(26,26))
        if activation=="gelu":
            self.activation=nn.GELU()
        else:
            raise ValueError("Invalid activation function")
        
        
        self.rotor_positions=[0,0,0]
        self.rotateMatrix=torch.arange(26).unsqueeze(1)

    def configure_optimizers(self):
    
        if self.optimizer_name=="adam":
            return torch.optim.Adam(self.parameters(),lr=self.learning_rate)
        elif self.optimizer_name=="sgd":
            return torch.optim.SGD(self.parameters(),lr=self.learning_rate)
        else:
            raise ValueError("Invalid optimizer name")

    def forward(self,x):
        #This is the logic for the enigma machine
        #We will generate a sequence of 50 random letters, then encode it.
        
        # x represents a batch of inputs, with shape (batch_size,length,26)
        
        #Step 1 : generate the rotation matrix for ALL positions 
        rotationSteps=torch.arange(x.shape[1]).unsqueeze(0) #This is a tensor of shape (1,length)
        rotationArrayForRotor1=torch.add(rotationSteps,self.rotateMatrix)%26
        rotationArrayForRotor2=torch.add(rotationSteps,self.rotateMatrix)//26
        rotationArrayForRotor3=torch.add(rotationSteps,self.rotateMatrix)//676

        # Step 2: make the one hot encoding of the rotors
        rotor1=torch.nn.functional.one_hot(rotationArrayForRotor1,26) #This is a tensor of shape (length,26,26)
        rotor2=torch.nn.functional.one_hot(rotationArrayForRotor2,26)
        rotor3=torch.nn.functional.one_hot(rotationArrayForRotor3,26)
        #Note, the above carries out the rotation of the rotors, with no gradient.

   


        #Step 3: each letter has to go through the rotors, and the reflector.
        rotor1=self.rotor_1@rotor1
        rotor2=self.rotor_2@rotor2
        rotor3=self.rotor_3@rotor3
        #To Do : Decide whether these 3 lines are necessary.
        rotor1=rotor1+1e-6
        rotor2=rotor2+1e-6
        rotor3=rotor3+1e-6
        #Finally, we have to apply the encoding
        x=torch.bmm(x,rotor1)
        x=torch.bmm(x,rotor2)
        x=torch.bmm(x,rotor3)
        x=self.reflector@x
        x=torch.bmm(x,rotor3)
        x=torch.bmm(x,rotor2)
        x=torch.bmm(x,rotor1)
        return x
    
    def training_step(self,batch,batch_idx):
        encoded,GT=batch #GT is the ground truth     -though the elegance of the enigma machine is that these can be the other way around and it will still work.
        decoded=self.forward(encoded)
        loss=self.loss(decoded,GT)
        self.log("loss",loss)
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
        current_rotor1=self.rotor_1
        current_rotor2=self.rotor_2
        current_rotor3=self.rotor_3
        rotor1Loss=torch.nn.functional.cross_entropy(current_rotor1,rotors[0])
        rotor2Loss=torch.nn.functional.cross_entropy(current_rotor2,rotors[1])
        rotor3Loss=torch.nn.functional.cross_entropy(current_rotor3,rotors[2])
        reflectorLoss=torch.nn.functional.cross_entropy(self.reflector,reflector)
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
    dm.setup()
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
    progress_bar = pl.callbacks.ProgressBar()


    trainer=pl.Trainer(max_epochs=10,
                        device="auto",
                        accelerator="auto",
                        logger=wandb_logger,
                        callbacks=[early_stop_callback,checkpoint_callback,progress_bar])
                        # have a look at the other flags that can be set here.

    
    trainer.fit(model,dm)

    model.compare_to_gt(dm.rotors,dm.reflector)

    print("Model Trained Successfully")


        



from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment as LSA
class Rotor(nn.Module):
    def __init__(self,position=1):
        super().__init__()
        self.rotor=nn.parameter.Parameter(torch.rand(26,26),requires_grad=True)
        nn.init.xavier_normal_(self.rotor)
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
        return torch.zeros(sequence_length,dtype=torch.long)
    def generate_rotation_matrixPos1(self,sequence_length):
        return torch.arange(sequence_length,dtype=torch.long)%26
    def generate_rotation_matrixPos2(self,sequence_length):
        return (torch.arange(sequence_length,dtype=torch.long)//26)+1 %26
    def generate_rotation_matrixPos3(self,sequence_length):
        return (torch.arange(sequence_length,dtype=torch.long)//676)+1 %26
    def forward(self,signalInput): 
        #signalInput is a tensor of shape (B,S,26) 
        Batch_Size=signalInput.shape[0]
        Sequence_Length=signalInput.shape[1]
        #Step 1: generate the rotation matrix
        rotationMatrix=torch.add(torch.arange(26,dtype=torch.long).unsqueeze(0),self.generate_rotation_matrix(Sequence_Length).unsqueeze(1))
        #Step 2: ensure the rotation matrix is <26
        rotationMatrix=rotationMatrix%26
        #plot the rotation matrix
        rotationMatrix=torch.nn.functional.one_hot(rotationMatrix,26).float()
        #Test - Add some noise to the rotation matrix?
        rotationMatrix=torch.matmul(rotationMatrix,(self.rotor/torch.norm(self.rotor,dim=(1),keepdim=True)).unsqueeze(0)).unsqueeze(0)

        #Test Consider Other ways of doing this? Maybe softmax? Do methods like Gumbel softmax work here?


        #shape is now (1, sequence_length,26,26)
        rotationMatrix=rotationMatrix.repeat(Batch_Size,1,1,1)
        #flatten the rotation matrix
        rotationMatrix=rotationMatrix.flatten(0,1)
        # Step 3: Apply the rotation matrix to the signal input
        # signalInput is of shape (B,S,26)
        signalInput = torch.bmm(signalInput.flatten(0,1).unsqueeze(1),rotationMatrix)
        #signalInput is now of shape (B*S,26,1)
        # Step 4: Remove the last dimension
        signalInput=signalInput.squeeze(1)
        signalInput=signalInput.unflatten(0,(Batch_Size,Sequence_Length))  

        #Test Add some sort of activation function here?
        signalInput=torch.nn.functional.softmax(signalInput,dim=2)
        return signalInput
    
    def reverse(self,signalInput): 
        #signalInput is a tensor of shape (B,S,26) 
        Batch_Size=signalInput.shape[0]
        Sequence_Length=signalInput.shape[1]
        rotationMatrix=torch.add(torch.arange(26,dtype=torch.long).unsqueeze(0),self.generate_rotation_matrix(Sequence_Length).unsqueeze(1))
        rotationMatrix=rotationMatrix%26
        rotationMatrix=torch.nn.functional.one_hot(rotationMatrix,26).float()
        rotationMatrix=torch.add(rotationMatrix,0.01*torch.randn_like(rotationMatrix))
        rotationMatrix=torch.matmul(rotationMatrix,(self.rotor/torch.norm(self.rotor,dim=1,keepdim=True)).unsqueeze(0)).unsqueeze(0)
        rotationMatrix=rotationMatrix.repeat(Batch_Size,1,1,1)
        rotationMatrix=rotationMatrix.flatten(0,1)
        rotationMatrix=rotationMatrix.permute(0,2,1) # This is the line that's different, reflecting that we're going the other direction through the rotor.
        signalInput = torch.bmm(signalInput.flatten(0,1).unsqueeze(1),rotationMatrix)
        signalInput=signalInput.squeeze(1)
        signalInput=signalInput.unflatten(0,(Batch_Size,Sequence_Length))  
        return signalInput
        


    

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
        elif activation=="tanh":
            self.activation=nn.Tanh()
        else:
            raise ValueError("Invalid activation function - Addd some more activation functions")
        self.save_hyperparameters()
    def configure_optimizers(self):
        params=list(self.R1.parameters()) + list(self.R2.parameters()) + list(self.R3.parameters()) + list(self.REF.parameters())
        if self.optimizer_name=="adam":
            return torch.optim.Adam(params,lr=self.learning_rate)
        elif self.optimizer_name=="sgd":
            return torch.optim.SGD(params,lr=self.learning_rate)
        
        #Test - Are there other optimizers that we can use? 
        else:
            raise ValueError("Invalid optimizer name")
        
        #To Do - See what schedulers do, and whether they are necessary. This function can return optimizers and schedulers. 

    def forward(self,x):
        # Test: Do we want an activation function here?
        x=self.activation(x)
        x=self.R1(x)
        x=self.activation(x)

        x=self.R2(x)
        x=self.activation(x)

        x=self.R3(x)
        x=self.activation(x)
        x=self.REF(x)
        x=self.activation(x)

        x=self.R1.reverse(x)
        x=self.activation(x)

        x=self.R2.reverse(x)

        x=self.activation(x)
        x=self.R3.reverse(x)
        x=self.activation(x)


        #Test: Do we want to do something like a softmax to force the gradient to a letter?
        x=torch.nn.functional.softmax(x,dim=2) ##what happens if this is removed? 
        return x
    
    def training_step(self,batch,batch_idx):
        encoded,GT=batch #GT is the ground truth     -though the elegance of the enigma machine is that these can be the other way around and it will still work. 

        GT=torch.nn.functional.one_hot(GT,26).permute(0,2,1).to(dtype=torch.float)
        encoded=torch.nn.functional.one_hot(encoded,26).to(dtype=torch.float)
        encoded.requires_grad=True
        #TEST: Does adding noise help? 
        encoded = encoded 
        decoded=self.forward(encoded)
        loss=self.loss(decoded.permute(0,2,1),GT)
        self.log("loss",loss,prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        #Test How close we are to the ground truth settings
        self.compare_to_gt(self.trainer.datamodule.dataset.rotors,self.trainer.datamodule.dataset.reflector)


    def print_enigma_settings(self):
        print("Rotor 1 : ",self.R1.rotor.max(1).indices)
        print("Rotor 2 : ",self.R2.rotor.max(1).indices)
        print("Rotor 3 : ",self.R3.rotor.max(1).indices)
        print("Reflector : ",self.REF.rotor.max(1).indices)
        print("Activation : ",self.activation)
        print("Optimizer : ",self.optimizer_name)
        print("Learning Rate : ",self.learning_rate)
        print("Batch Size : ",self.batch_size)
        print("Precision : ",self.precision)
        print("Activation : ",self.activation)

    def compare_to_gt(self,rotors,reflector):
        #This function will take a batch of ground truth rotors and reflectors, and compare them to the current settings.
        rotor1Loss=torch.nn.functional.cross_entropy(self.R1.rotor/torch.norm(self.R1.rotor,dim=(0,1),keepdim=True),rotors[0]) 
        rotor2Loss=torch.nn.functional.cross_entropy(self.R2.rotor/torch.norm(self.R2.rotor,dim=(0,1),keepdim=True),rotors[1])
        rotor3Loss=torch.nn.functional.cross_entropy(self.R3.rotor.T/torch.norm(self.R3.rotor,dim=(0,1),keepdim=True),rotors[2])
        reflectorLoss=torch.nn.functional.cross_entropy(self.REF.rotor/torch.norm(self.REF.rotor,dim=(0,1),keepdim=True),reflector)
        self.log("Rotor 1 Loss",rotor1Loss, prog_bar=True)
        self.log("Rotor 2 Loss",rotor2Loss, prog_bar=True) 
        self.log("Rotor 3 Loss",rotor3Loss, prog_bar=True)
        self.log("Reflector Loss",reflectorLoss, prog_bar=True)
        self.log("Test Loss",(rotor1Loss+rotor2Loss+rotor3Loss+reflectorLoss)/4,prog_bar=True)
        #use imshow to display the rotors and reflectors
        fig=plt.figure()
        ax1=fig.add_subplot(221)
        ax1.imshow(self.R1.rotor.detach().numpy())
        ax1.set_title("Rotor 1")
        ax2=fig.add_subplot(222)
        ax2.imshow(self.R2.rotor.detach().numpy())
        ax2.set_title("Rotor 2")
        ax3=fig.add_subplot(223)
        ax3.imshow(self.R3.rotor.detach().numpy())
        ax3.set_title("Rotor 3")
        ax4=fig.add_subplot(224)
        ax4.imshow(self.REF.rotor.detach().numpy())
        ax4.set_title("Reflector")
        #save to figure and log it
        #self.logger.experiment.log({"Rotor 1":fig})
        fig.savefig("Rotors.png")
        plt.close(fig)
        # self.logger.experiment.log({"Rotors":wandb.Image("Rotors.png")})

        LSA_score=self.convertParametertoConfidence(self.R1.rotor)
        self.log("Rotor 1 Confidence",LSA_score)
        LSA_score=self.convertParametertoConfidence(self.R2.rotor)
        self.log("Rotor 2 Confidence",LSA_score)
        LSA_score=self.convertParametertoConfidence(self.R3.rotor)
        self.log("Rotor 3 Confidence",LSA_score)
        LSA_score=self.convertParametertoConfidence(self.REF.rotor)
        self.log("Reflector Confidence",LSA_score)


        # print("Mean Solution loss : ",(rotor1Loss+rotor2Loss+rotor3Loss+reflectorLoss)/4)
      
        return rotor1Loss,rotor2Loss,rotor3Loss,reflectorLoss

        #We will compare the current settings to the ground truth settings.

    def convertParametertoConfidence(self, param):
        #This function will take a tensor of shape (26,26) and convert it to a tensor of shape (26,26) where each row is a probability distribution.
        #This is done by normalizing the tensor along the 0th axis.
        cost_matrix=(param/torch.norm(param,dim=(0,1),keepdim=True)).detach().numpy()
        col_ind,row_ind=LSA(cost_matrix,maximize=True)
        return cost_matrix[row_ind, col_ind].sum()


if __name__ == "__main__":
    #build arguments
    import argparse
    import pytorch_lightning as pl
    from DataModule import EnigmaDataModule
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer_name",type=str,default="sgd")
    parser.add_argument("--learning_rate",type=float,default=1e-3)
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--precision",type=str,default="32")
    parser.add_argument("--activation",type=str,default="sigmoid")
    parser.add_argument("--loss",type=str,default="CrossEntropy")
    args=parser.parse_args()
    model=Enigma(args.optimizer_name,args.learning_rate,args.batch_size,args.precision,args.activation,args.loss)
    model.print_enigma_settings()

    dm=EnigmaDataModule()
    dm.setup(stage="train")
   
    #login as anonymous
    wandb.login(anonymous="allow")
    wandb_logger = WandbLogger(project='Enigma', entity='st7ma784')

    #enable Early stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='Test Loss',
        patience=4,
        verbose=False,
        mode='min'
    )

    # #enable model checkpointing
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     monitor='loss',
    #     dirpath='./',
    #     filename='Enigma-{epoch:02d}-{loss:.2f}',
    #     save_top_k=3,
    #     mode='min',
    # )

    #And finally a nice loading bar
    progress_bar = pl.callbacks.TQDMProgressBar()


    trainer=pl.Trainer(max_epochs=500,
                       max_steps=30000,
                        devices="auto",
                        accelerator="auto",
                        logger=wandb_logger,
                        callbacks=[early_stop_callback,progress_bar])
                        # have a look at the other flags that can be set here.

    
    trainer.fit(model,dm)

    
    print("Model Trained Successfully")


        

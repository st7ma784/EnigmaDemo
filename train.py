

from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment as LSA


#This is the rotor class. It will take a signal input, and rotate it by a certain amount.
#The rotor class will have a forward and reverse function.

#HACKER TASK: Have a look at how this repo is training assignment matrices - and copy into the rotor class!  https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch/blob/master/my_sorting_train.py 
#             A better walkthrough of the ideas are here : https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/permutations.html


class RotorBase(LightningModule):
    def __init__(self,position=1):
        super().__init__()
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
        self.rotor=nn.parameter.Parameter(torch.rand(26,26),requires_grad=True)
        nn.init.xavier_normal_(self.rotor) 
        self.rotor.data=self.rotor.data/torch.norm(self.rotor.data,dim=1,keepdim=True) 

    def getWeight(self):
        return self.rotor.to(self.device)
    def generate_rotation_matrixPos0(self,sequence_length):
        return torch.zeros(sequence_length,dtype=torch.long)
    def generate_rotation_matrixPos1(self,sequence_length):
        return torch.arange(sequence_length,dtype=torch.long)%26
    def generate_rotation_matrixPos2(self,sequence_length):
        return (torch.arange(sequence_length,dtype=torch.long)//7) %26
    def generate_rotation_matrixPos3(self,sequence_length):
        return (torch.arange(sequence_length,dtype=torch.long)//11) %26

    def make_rotation_matrix(self,signalInput):
        Sequence_Length=signalInput.shape[1]
        # Step 1 Build a rotation matrix, that is a tensor of shape (S,26,26) that will rotate the so that in position 1 , its identity, in position 2, it's rotated by 1, etc.
        rotationMatrix=self.generate_rotation_matrix(Sequence_Length)
        rotationMatrix=torch.nn.functional.one_hot(rotationMatrix,26).float().to(self.device) # This is a tensor of shape (S,26) which, we 
        rotationMatrix= torch.bmm(rotationMatrix.unsqueeze(2),rotationMatrix.unsqueeze(1)) 
        # Step 2: For each value of s, [b,h]@[h,h]
        return torch.bmm(signalInput.permute(1,0,2),rotationMatrix.permute(0,2,1)).permute(1,0,2)
        
    def forward(self,signalInput): 
        signalInput=self.make_rotation_matrix(signalInput)
        signalInput=signalInput@(self.rotor)
        #Test Add some sort of activation function here?
        return signalInput
    
    def reverse(self,signalInput): 
        signalInput=self.make_rotation_matrix(signalInput)
        signalInput=signalInput@(self.rotor.T)
        #Test Add some sort of activation function here?
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
        if activation=="gelu":
            self.activation=nn.GELU()
        elif activation=="relu":
            self.activation=nn.ReLU()
        elif activation=="tanh":
            self.activation=nn.Tanh()
        elif activation=="sigmoid":
            self.activation=nn.Sigmoid()
        elif activation=="softplus":
            self.activation=nn.Softplus()
        else:
            raise ValueError("Invalid activation function")
        if lossName=="CrossEntropy":
            self.loss=nn.CrossEntropyLoss(reduction="mean")
            self.process= self.processNullLabels
        elif lossName=="BCELoss":
            self.loss=nn.BCELoss(reduction="mean")
            self.process=self.processNullLabels
        elif lossName=="MSELoss":
            self.loss=nn.MSELoss(reduction="mean")
            self.process=self.processLabels
        elif lossName=="DICE":
            self.loss=nn.DiceLoss(reduction="mean")
            self.process=self.processLabels
        else:
            raise ValueError("Invalid loss function")
        
        self.R1=RotorBase(1)
        self.R2=RotorBase(2)
        self.R3=RotorBase(3)
        self.REF=RotorBase(0)
        #move all the rotors to the device

        #make sure these are all on device! 

        self.softmax=nn.Softmax(dim=2)

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

    def processLabels(self,labels):
        #This function will take a tensor of shape (B,S) and convert it to a tensor of shape (B,S,26) where each row is a probability distribution.
        return torch.nn.functional.one_hot(labels,26).permute(0,2,1).to(dtype=torch.float)
    
    def processNullLabels(self,labels):
        return labels
    
    def forward(self,x):
        # Test: Do we want an activation function here?
        x=self.R1(x)
        x=self.R2(x)
        x=self.R3(x)
        x=self.REF(x)
        x=self.R1.reverse(x)
        x=self.R2.reverse(x)
        x=self.R3.reverse(x)
        x=self.softmax(x)
        #Test: Do we want to do something like a softmax to force the gradient to a letter?
        #Test: could try learning these one at a time then freezing them? 
        return x
    
    def training_step(self,batch,batch_idx):
        encoded,GT=batch #GT is the ground truth     -though the elegance of the enigma machine is that these can be the other way around and it will still work. 
        GT=self.process(GT).to(self.device)
        
        encoded=torch.nn.functional.one_hot(encoded,26,).to(dtype=torch.float)
        encoded=encoded.to(self.device)

        encoded.requires_grad=True
        #TEST: Does adding noise help? 
        decoded=self.forward(encoded) 
        loss=self.loss(decoded.permute(0,2,1),GT)

        #Test: Do we want to add a loss for the rotors and reflectors? Maybe a cross entropy loss? How about using the norms of the matrices?
        self.log( self.lossName,loss,prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        self.compare_to_gt(self.trainer.datamodule.dataset.rotors,self.trainer.datamodule.dataset.reflector)


    def print_enigma_settings(self):
        print("Rotor 1 : ",self.R1.getWeight().max(1).indices)
        print("Rotor 2 : ",self.R2.getWeight().max(1).indices)
        print("Rotor 3 : ",self.R3.getWeight().max(1).indices)
        print("Reflector : ",self.REF.getWeight().max(1).indices)
        print("Activation : ",self.activation)
        print("Optimizer : ",self.optimizer_name)
        print("Learning Rate : ",self.learning_rate)
        print("Batch Size : ",self.batch_size)
        print("Precision : ",self.precision)
        print("Activation : ",self.activation)

    def compare_to_gt(self,rotors,reflector):
        #This function will take a batch of ground truth rotors and reflectors, and compare them to the current settings.
        rotor1Loss=torch.nn.functional.cross_entropy(self.R1.getWeight()/torch.norm(self.R1.getWeight(),dim=(1),keepdim=True),rotors[0].to(self.device)) 
        rotor2Loss=torch.nn.functional.cross_entropy(self.R2.getWeight()/torch.norm(self.R2.getWeight(),dim=(1),keepdim=True),rotors[1].to(self.device))
        rotor3Loss=torch.nn.functional.cross_entropy(self.R3.getWeight()/torch.norm(self.R3.getWeight(),dim=(1),keepdim=True),rotors[2].to(self.device))
        reflectorLoss=torch.nn.functional.cross_entropy(self.REF.getWeight()/torch.norm(self.REF.getWeight(),dim=(0,1),keepdim=True),reflector.to(self.device))
        rotor1LossT=torch.nn.functional.cross_entropy(self.R1.getWeight().T/torch.norm(self.R1.getWeight(),dim=(1),keepdim=True),rotors[0].to(self.device)) 
        rotor2LossT=torch.nn.functional.cross_entropy(self.R2.getWeight().T/torch.norm(self.R2.getWeight(),dim=(1),keepdim=True),rotors[1].to(self.device))
        rotor3LossT=torch.nn.functional.cross_entropy(self.R3.getWeight().T/torch.norm(self.R3.getWeight(),dim=(1),keepdim=True),rotors[2].to(self.device))
        reflectorLossT=torch.nn.functional.cross_entropy(self.REF.getWeight().T/torch.norm(self.REF.getWeight(),dim=(0,1),keepdim=True),reflector.to(self.device))
        TestLoss=(min(rotor1Loss.item(),rotor1LossT.item())+min(rotor2Loss.item(),rotor2LossT.item())+min(rotor3Loss.item(),rotor3LossT.item())+min(reflectorLoss.item(),reflectorLossT.item()))/4
        self.log("Rotor 1 Loss",min(rotor1Loss.item(),rotor1LossT.item()))
        self.log("Rotor 2 Loss",min(rotor2Loss.item(),rotor2LossT.item()))
        self.log("Rotor 3 Loss",min(rotor3Loss.item(),rotor3LossT.item()))
        self.log("Reflector Loss",min(reflectorLoss.item(),reflectorLossT.item()))
        self.log("Test Loss",TestLoss, prog_bar=True)
        #use imshow to display the rotors and reflectors
        fig=plt.figure()
        ax1=fig.add_subplot(241)
        ax1.imshow(self.R1.getWeight().detach().cpu().numpy())
        ax2=fig.add_subplot(242)
        ax2.imshow(self.R2.getWeight().detach().cpu().numpy())
        ax3=fig.add_subplot(243)
        ax3.imshow(self.R3.getWeight().detach().cpu().numpy())
        ax4=fig.add_subplot(244)
        ax4.imshow(self.REF.getWeight().detach().cpu().numpy())
        #plot original rotors
        ax5=fig.add_subplot(245)
        ax5.imshow(torch.nn.functional.one_hot(rotors[0]).detach().cpu().numpy())
        ax6=fig.add_subplot(246)
        ax6.imshow(torch.nn.functional.one_hot(rotors[1]).detach().cpu().numpy())
        ax7=fig.add_subplot(247)
        ax7.imshow(torch.nn.functional.one_hot(rotors[2]).detach().cpu().numpy())
        ax8=fig.add_subplot(248)
        ax8.imshow(torch.nn.functional.one_hot(reflector).detach().cpu().numpy())

        ax1.set_title("Rotor 1") 
        ax2.set_title("Rotor 2")
        ax3.set_title("Rotor 3")
        ax4.set_title("Reflector")
        ax5.set_title("Rotor 1 GT")
        ax6.set_title("Rotor 2 GT")
        ax7.set_title("Rotor 3 GT")
        ax8.set_title("Reflector GT")

        ax1.axis("off")
        ax2.axis("off")
        ax3.axis("off")
        ax4.axis("off")
        ax5.axis("off")
        ax6.axis("off")
        ax7.axis("off")
        ax8.axis("off")

        fig.savefig("Rotors.currentEpoch{}.TestLoss{}.png".format(self.current_epoch,TestLoss))
        plt.close(fig)
        self.LogParametertoConfidence("Rotor 1 Confidence", self.R1.getWeight())
        self.LogParametertoConfidence("Rotor 2 Confidence", self.R2.getWeight())
        self.LogParametertoConfidence("Rotor 3 Confidence", self.R3.getWeight())
        self.LogParametertoConfidence("Reflector Confidence", self.REF.getWeight())


    def LogParametertoConfidence(self, name,param):
        #This function will take a tensor of shape (26,26) and convert it to a tensor of shape (26,26) where each row is a probability distribution.
        cost_matrix=(param/ (26*torch.norm(param,dim=1,keepdim=True))).cpu().detach().numpy()
        col_ind,row_ind=LSA(cost_matrix,maximize=True)
        self.log(name,cost_matrix[row_ind, col_ind].sum())


if __name__ == "__main__":
    #build arguments
    import argparse
    import pytorch_lightning as pl
    from DataModule import EnigmaDataModule
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer_name",type=str,default="sgd")
    parser.add_argument("--learning_rate",type=float,default=0.2)
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--precision",type=str,default="32")
    parser.add_argument("--activation",type=str,default="gelu")
    parser.add_argument("--loss",type=str,default="MSELoss")
    args=parser.parse_args()
    model=Enigma(args.optimizer_name,args.learning_rate,args.batch_size,args.precision,args.activation,args.loss)
    model.print_enigma_settings()

    dm=EnigmaDataModule()
    dm.setup(stage="train")
   
    #login as anonymous
    wandb.login(anonymous="allow")
    wandb_logger = WandbLogger(project='Enigma', entity='st7ma784')

    #And finally a nice loading bar
    progress_bar = pl.callbacks.TQDMProgressBar()
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



    trainer=pl.Trainer(max_epochs=500,
                       max_steps=30000,
                        devices="auto",
                        accelerator="auto",
                        logger=wandb_logger,
                        callbacks=[early_stop_callback,progress_bar])
                        # have a look at the other flags that can be set here.

    
    trainer.fit(model,dm)

    
    print("Model Trained Successfully")


        

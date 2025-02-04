

from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment as LSA

from pytorchGSTest import gumbel_sinkhorn
#This is the rotor class. It will take a signal input, and rotate it by a certain amount.
#The rotor class will have a forward and reverse function.

#HACKER TASK: Have a look at how this repo is training assignment matrices - and copy into the rotor class!  https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch/blob/master/my_sorting_train.py 
#             A better walkthrough of the ideas are here : https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/permutations.html


def RotorHook(self, input, output):
    #This function will be called whenever the rotor is called. 
    #We will use this to add noise to the rotor. 
    fig=plt.figure()
    ax1=fig.add_subplot(131)
    ax1.imshow(input[0][0].cpu().detach().numpy())
    ax1.set_title("input")
    ax2=fig.add_subplot(132)
    ax2.imshow(self.rotor.cpu().detach().numpy())
    ax2.set_title("Rotor")
    ax3=fig.add_subplot(133)
    ax3.imshow(output[0].cpu().detach().numpy())
    ax3.set_title("Output")
    fig.savefig("RotorHookPosition{}.png".format(self.position))
    plt.close(fig)

class RotorBase(LightningModule):
    def __init__(self,position=1,gs_tau=0.1,gs_n_iter=50,gs_noise_factor=0.2):
        super().__init__()
        self.position=position
        self.generate_rotation_matrix=[self.generate_rotation_matrixPos0,self.generate_rotation_matrixPos1,self.generate_rotation_matrixPos2,self.generate_rotation_matrixPos3][position]
        self.rotor=nn.parameter.Parameter(torch.rand(26,26),requires_grad=True)
        nn.init.xavier_normal_(self.rotor) 
        self.rotor.data=gumbel_sinkhorn(self.rotor.data,gs_tau,gs_n_iter,gs_noise_factor)
        self.gs_tau=gs_tau
        self.gs_n_iter=gs_n_iter
        self.gs_noise_factor=gs_noise_factor
        self.Base=torch.arange(26).unsqueeze(0)
        self.ProcessPermute=self.ForwardMatrix
        if self.gs_n_iter==0:
            self.getrotor=self.Rotor
        else :
            self.getrotor=self.GSRotor
    def rotorForward(self):
        self.ProcessPermute=self.ForwardMatrix
    def rotorReverse(self):
        self.ProcessPermute=self.ReverseMatrix
    def getWeight(self):
        return self.rotor.to(self.device)
    def ReverseMatrix(self,x):
        return x.permute(0,2,1)
    def ForwardMatrix(self,x):
        return x
    def generate_rotation_matrixPos0(self,sequence_length):
        return torch.zeros(sequence_length,dtype=torch.long)
    def generate_rotation_matrixPos1(self,sequence_length):
        return torch.remainder(torch.arange(sequence_length,dtype=torch.long),26)
    def generate_rotation_matrixPos2(self,sequence_length):
        return torch.remainder(torch.arange(sequence_length,dtype=torch.long)//7,26)
    def generate_rotation_matrixPos3(self,sequence_length):
        return torch.remainder(torch.arange(sequence_length,dtype=torch.long)//11,26)
    def Rotor(self):
        return self.rotor
    def GSRotor(self):
        return gumbel_sinkhorn(self.rotor.data,self.gs_tau,self.gs_n_iter,self.gs_noise_factor)
    def forward(self,signalInput):
        Sequence_Length=signalInput.shape[1]
        # Step 1 Build a rotation matrix, that is a tensor of shape (S,26,26) that will rotate the so that in position 1 , its identity, in position 2, it's rotated by 1, etc.
        rotationMatrix=self.generate_rotation_matrix(Sequence_Length).unsqueeze(1)
        rotationMatrix=torch.remainder(torch.add(rotationMatrix,self.Base),26)
        positionRotorMatrix=torch.nn.functional.one_hot(rotationMatrix,26).to(torch.float) #this is an offset identity matrix of shape (S,26,26)
        # Step 2: Multiply the rotor by the offset identity matrix
        positionRotorMatrix=positionRotorMatrix@self.getrotor() #this should be s,26,26
        # Step 3: Multiply the signal by the position rotor matrix
        positionRotorMatrix=self.ProcessPermute(positionRotorMatrix)
        signalOutput=torch.bmm(signalInput.permute(1,0,2),positionRotorMatrix).permute(1,0,2)
        return signalOutput
    
 
class Reflector(RotorBase):
    def forward(self,signalInput):
        return signalInput@self.rotor
    
class Enigma(LightningModule):

    def __init__(self,optimizer_name,learning_rate,batch_size,precision,activation,lossName="CrossEntropy",gs_tau=0.1,gs_n_iter=50,gs_noise_factor=0.2):
        super().__init__()
        self.optimizer_name=optimizer_name
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.precision=precision
        self.activation=activation
        self.lossName=lossName
        self.gs_tau=gs_tau
        self.gs_n_iter=gs_n_iter
        self.gs_noise_factor=gs_noise_factor
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
        
        self.R1=RotorBase(1,gs_tau,gs_n_iter,gs_noise_factor)
        self.R2=RotorBase(2,gs_tau,gs_n_iter,gs_noise_factor)
        self.R3=RotorBase(3,gs_tau,gs_n_iter,gs_noise_factor)
        self.REF=Reflector(0,gs_tau,gs_n_iter,gs_noise_factor)
        self.softmax=nn.Softmax(dim=2)
        L1=torch.nn.L1Loss()
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
    
    def reverseRotors(self):
        self.R1.rotorReverse()
        self.R2.rotorReverse()
        self.R3.rotorReverse()
    
    def forwardRotors(self):
        self.R1.rotorForward()
        self.R2.rotorForward()
        self.R3.rotorForward()

    def forward(self,x):
        # Test: Do we want an activation function here?

        self.forwardRotors()
        
        x=self.R1(x)
        x=self.activation(x)
        x=self.R2(x)
        x=self.R3(x)
        x=self.activation(x)
        x=self.REF(x)
        self.reverseRotors()
        x=self.R3(x)
        x=self.activation(x)
        x=self.R2(x)
        x=self.R1(x)
        x=self.softmax(x)
        #Test: Do we want to do something like a softmax to force the gradient to a letter?
        #Test: could try learning these one at a time then freezing them? 
        return x
    
    def training_step(self,batch,batch_idx):
        encoded,GT=batch #GT is the ground truth     -though the elegance of the enigma machine is that these can be the other way around and it will still work. 
        GT=self.process(GT).to(self.device)
        encoded=encoded.to(self.device)
        encoded=torch.nn.functional.one_hot(encoded,26,).to(dtype=torch.float)
        #TEST: Does adding noise help? 
        decoded=self.forward(encoded) 
        loss=self.loss(decoded.permute(0,2,1),GT)
        self.log(self.lossName,loss,prog_bar=True)
        #L1 Loss will be the rowwise sum of the rotors.
        #L2 will be the rowwise sum of the square of the rotors.
        L2=torch.abs(1-torch.norm(self.R1.getWeight(),dim=(1),keepdim=True).mean()) + torch.abs(1-torch.norm(self.R2.getWeight(),dim=(1),keepdim=True).mean()) + torch.abs(1-torch.norm(self.R3.getWeight(),dim=(1),keepdim=True).mean()) + torch.abs(1-torch.norm(self.REF.getWeight(),dim=(0),keepdim=True).mean())
        L2=L2/4
        self.log("L2Loss", torch.abs(1-L2))
        L1=torch.abs(1-torch.norm(self.R1.getWeight(),p=1,dim=(1),keepdim=True).mean()) + torch.abs(1-torch.norm(self.R2.getWeight(),p=1,dim=(1),keepdim=True).mean()) + torch.abs(1-torch.norm(self.R3.getWeight(),p=1,dim=(1),keepdim=True).mean()) + torch.abs(1-torch.norm(self.REF.getWeight(),p=1,dim=(0),keepdim=True).mean())
        L1=L1/4
        self.log("L1Loss", torch.abs(1-L1))

        '''
        We can also do some logic here around the use of "crabs" and "lobsters" in enigma, where we can isolate repeated characters in the input and output, within short spans of each other, and use it to boost the loss gradient in a certain direction! 

        Step 1: Find the rotation matrices, and find, for each rotor, the index of positions in the whole sequence whre only that rotor is in a different position.
        Step 2: For each set of indexes, check the sequence for pairs of input+output that are the same or reverses of each other. 
        step3: for each of those pairs, calculate the distance apart they are. 
        Step 4: that distance means that where the rotor in position x might have X->Y, the rotor in position x+distance might have Y->X.  
        step 5: This means we can make a matrix knowing that from a certain input on that rotor, that rotors parameters are offset by "distance" in the row "distance" down the rotor. 
        Step 6: We can then use the L1 loss to push the rotor assignments to be more like the row "distance" down the rotor.
        step 7: sum this loss across all occurrences of the rotor in that position.     
        
        '''


        return loss#--+L2+L1
    
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
        L=torch.nn.CrossEntropyLoss()
        rotor1Loss=L(self.R1.getWeight()/torch.norm(self.R1.getWeight(),dim=(1),keepdim=True),rotors[0].to(self.device)) 
        rotor2Loss=L(self.R2.getWeight()/torch.norm(self.R2.getWeight(),dim=(1),keepdim=True),rotors[1].to(self.device))
        rotor3Loss=L(self.R3.getWeight()/torch.norm(self.R3.getWeight(),dim=(1),keepdim=True),rotors[2].to(self.device))
        reflectorLoss=L(self.REF.getWeight()/torch.norm(self.REF.getWeight(),dim=(0),keepdim=True),reflector.to(self.device))
        rotor1LossT=L(self.R1.getWeight().T/torch.norm(self.R1.getWeight(),dim=(1),keepdim=True),rotors[0].to(self.device)) 
        rotor2LossT=L(self.R2.getWeight().T/torch.norm(self.R2.getWeight(),dim=(1),keepdim=True),rotors[1].to(self.device))
        rotor3LossT=L(self.R3.getWeight().T/torch.norm(self.R3.getWeight(),dim=(1),keepdim=True),rotors[2].to(self.device))
        reflectorLossT=L(self.REF.getWeight().T/torch.norm(self.REF.getWeight(),dim=(0),keepdim=True),reflector.to(self.device))
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
    parser.add_argument("--gs_tau",type=float,default=10)
    parser.add_argument("--gs_n_iter",type=int,default=10)
    parser.add_argument("--gs_noise_factor",type=float,default=0.5)
    parser.add_argument("--loss",type=str,default="CrossEntropy")
    parser.add_argument("--debug",action="store_true")
    args=parser.parse_args()

    if args.debug:
        args.gs_tau=1
        args.gs_n_iter=0
        args.gs_noise_factor=0
    model=Enigma(args.optimizer_name,args.learning_rate,args.batch_size,args.precision,args.activation,args.loss,args.gs_tau,args.gs_n_iter,args.gs_noise_factor)
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

    if args.debug:
        trainer=pl.Trainer(fast_dev_run=True,
                        callbacks=[early_stop_callback,progress_bar])
        model.R1.rotor.data=torch.nn.functional.one_hot(dm.rotors[0]).float()
        model.R1.register_forward_hook(RotorHook)
        model.R2.rotor.data=torch.nn.functional.one_hot(dm.rotors[1]).float()
        model.R2.register_forward_hook(RotorHook)
        model.R3.rotor.data=torch.nn.functional.one_hot(dm.rotors[2]).float()
        model.R3.register_forward_hook(RotorHook)
        model.REF.rotor.data=torch.nn.functional.one_hot(dm.reflector).float()
        model.REF.register_forward_hook(RotorHook)
        input=torch.randint(0,26,(1,50),dtype=torch.long)
        input=torch.nn.functional.one_hot(input,26).float()
        #Use property of enigma that the output of the enigma machine is the can be reentered to decode the message. to check if it is bidirectional
        output=model(input)
        output2=model(output)
        if not torch.allclose(input,output2):
            print("=====================================")
            print("Enigma Machine is not working as bidirectional")
            print("=====================================")
            fig=plt.figure()
            ax1=fig.add_subplot(131)
            ax1.imshow(input[0].cpu().detach().numpy())
            ax1.set_title("input")
            ax2=fig.add_subplot(132)
            ax2.imshow(output[0].cpu().detach().numpy())
            ax2.set_title("Output")
            ax3=fig.add_subplot(133)
            ax3.imshow(output2[0].cpu().detach().numpy())
            ax3.set_title("Output2 - should be the same as input")
            fig.savefig("EnigmaTest.png")
            plt.close(fig)
            
    else:
        trainer=pl.Trainer(max_epochs=500,
                       max_steps=30000,
                        devices="auto",
                        accelerator="auto",
                        logger=wandb_logger,
                        callbacks=[early_stop_callback,progress_bar])
                        # have a look at the other flags that can be set here.

    
    trainer.fit(model,dm)

    
    print("Model Trained Successfully")


        

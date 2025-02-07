import wandb
wandb.login(anonymous="allow")

# Create a sweep configuration as found at https://docs.wandb.ai/guides/sweeps/define-sweep-configuration/
if __name__=="__main__":
    sweep_config = {
        'method': 'bayes',  # Randomly sample the hyperparameter space (alternatives: grid, random)
        'program': 'train.py',  # This is the script to be run
    
        'metric': {  # This is the metric we are interested in maximizing
            'name': 'Test Loss',
            'goal': 'minimize'   
        },
        'parameters': {
            'learning_rate': {
                'values':[1e-1,1e-2,1e-3,2e-4,1e-4,5e-5,2e-5]
            },
            'batch_size': {
                'values': [256]
            },
            'precision': {
                'values': ['32']
            },
            'activation':{
                'values': ['gelu','relu','tanh','sigmoid']
            },
            "gs_tau":{
                "values":[0.01, 0.03, 0.05, 0.1, 0.5]
            },
            "gs_n_iter":{
                "values":[1,10,50,100]
            },
            "gs_noise_factor":{
                "values":[0,0.2,0.4,0.6,0.8]
            },
            "optimizer_name":{
                "values":["adam","sgd"]
            },
            "loss":{
                "values":["CrossEntropy","MSELoss"]
            }
        }
    }

    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project="Enigma", entity="st7ma784")
    import os
    # Initialize a new wandb agent and run the train.py script with the sweep_id args
    os.system(f"wandb agent {sweep_id} --count 100")
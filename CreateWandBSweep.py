import wandb
wandb.login()
if __name__=="__main__":
    sweep_config = {
        'method': 'random',  # Randomly sample the hyperparameter space (alternatives: grid, bayes)
        'metric': {  # This is the metric we are interested in maximizing
            'name': 'loss',
            'goal': 'minimize'   
        },
        'parameters': {
            'learning_rate': {
                'values':[2e-4,1e-4,5e-5,2e-5]
            },
            'batch_size': {
                'values': [16,24,32]
            },
            'precision': {
                'values': ['32']
            },
            'activation':{
                'values': ['gelu']
            },
            "optimizer_name":{
                "values":["adam","sgd"]
            },
            "loss":{
                "values":["CrossEntropy","MSE"]
            }
        }
    }

    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project="Enigma", entity="st7ma784")
    print(sweep_id)

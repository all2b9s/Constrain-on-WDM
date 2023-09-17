import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import colors

import torch
from torch import nn
from torch import optim

import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import os
import optuna
from optuna.trial import TrialState
import os

device=torch.device(1 if torch.cuda.is_available() else 'cpu')
print(device)

BATCH_SIZE=1024
N_VARIABLE = 14+5
N_CONTEXT = 1

train_data = torch.load('./data/all_in_1_train.pt',map_location=device)
test_data = torch.load('./data/all_in_1_test.pt',map_location=device)
valid_data = torch.load('./data/all_in_1_valid.pt',map_location=device)

def normalizing(samples, mean, std):
    n = samples.size(0)
    result = torch.zeros(samples.size())
    for i in range(n):
        result[i] = (samples[i]-mean[i])/std[i]
    return result

BATCH_SIZE=1024

all_data = torch.cat([train_data,valid_data,test_data], dim=1)
all_mean = torch.std_mean(all_data,dim=1)[1]
all_std = torch.std_mean(all_data,dim=1)[0]


norm_train = torch.t(normalizing(train_data,all_mean,all_std)).to(device)
norm_test = torch.t(normalizing(test_data,all_mean,all_std)).to(device)
norm_valid = torch.t(normalizing(valid_data,all_mean,all_std)).to(device)


def define_model(trial):
    
# Create the reverse model
    from pyro.nn import ConditionalAutoRegressiveNN
    num_layers = trial.suggest_int("num_layers", 1, 10)
    input_dim = N_CONTEXT
    context_dim = N_VARIABLE
    count_bins = trial.suggest_int("count_bins", 2, 64)
    #hidden_dims=[256,128]
    hidden_dims= [trial.suggest_int("hidden_dims", 32, 256,log=True)]
    bound = trial.suggest_int("bound", 2, 10)

    spline_dims = [count_bins, count_bins, count_bins - 1, count_bins]

    base_dist = dist.Normal(torch.zeros(input_dim).to(device), torch.ones(input_dim).to(device))
    transform = []

    for i in range(0,num_layers):
        hypernet = ConditionalAutoRegressiveNN(input_dim, hidden_dims=hidden_dims,
                                                context_dim=context_dim,
                                                nonlinearity=torch.nn.ReLU(),
                                                param_dims=spline_dims)

        transform.append(T.ConditionalSplineAutoregressive(input_dim, hypernet,
                                                                    bound=bound, order='linear',
                                                                    count_bins=count_bins).to(device))
        
    flow = dist.ConditionalTransformedDistribution(base_dist, transform)

    return flow,transform


def objective(trial):
    train_dataset = torch.utils.data.TensorDataset(norm_train[:,:N_VARIABLE], norm_train[:,N_VARIABLE:])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # Generate the model.
    flow,transform = define_model(trial)
    
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    weight_decay = trial.suggest_float("weight_decay", 1e-5, 0.1, log=True)

    optimizer = optim.Adam((par for model in transform for par in model.parameters()),lr = lr,weight_decay = weight_decay )

    min_valid = 1e40
    fmodel = './models/WDM/all_in_new_%d.pt'%(trial.number)
    log_name = './data/Trial_log/all_in_new_%d.txt'%(trial.number)

    num_iter = 40
    lr = 1e-4
    weight_decay = 1e-3

    optimizer = optim.Adam((par for model in transform for par in model.parameters()),lr = lr,weight_decay = weight_decay )
    log = []
    for step in range(0,40):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = -flow.condition(x.detach()).log_prob(y.detach()).mean()
            loss.backward()
            optimizer.step()
            flow.clear_cache()
        
        with torch.no_grad():
            error = -flow.condition(norm_valid[:,:N_VARIABLE]).log_prob(norm_valid[:,N_VARIABLE:]).mean()
            #print(loss,error)
        
        if error<min_valid:  
            min_valid = error
            torch.save(flow, fmodel)

        log.append([loss.to('cpu').detach().numpy(),error.to('cpu').detach().numpy()])
        trial.report(min_valid,step)
        
        
        # Handle pruning based on the intermediate value.
        #if trial.should_prune():
            #raise optuna.exceptions.TrialPruned()
    log = np.array(log)
    np.savetxt(log_name,log)
    return min_valid


def main():
    #optuna.delete_study(study_name='Omega_var',storage='sqlite:///Omega_var.db')
    study = optuna.create_study(direction="minimize",study_name='all_in_new_1',storage='sqlite:///all_in_new_1.db',load_if_exists=True)
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(df)
    study.optimize(objective, n_trials=200,timeout = 3600*24)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    main()
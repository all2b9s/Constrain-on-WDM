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



class CAMELS():
    BATCH_SIZE = 512
    def __init__(self):
        M_star = torch.tensor([])
        Met_star = torch.tensor([])
        M_gas = torch.tensor([])
        Met_gas = torch.tensor([])
        WDM_mass = torch.tensor([])

        context =np.loadtxt('/home/shurui/CAMELS/sobol_sequence_WDM_real_values.txt',usecols=[5])

        self.data = [M_star,Met_star,M_gas,Met_gas,WDM_mass]
        for num in range(0,1024):
            
            
            # catalogue name
            catalogue = '/home/shurui/CAMELS/WDM/WDM_'+str(num)+'/fof_subhalo_tab_090.hdf5'

            # value of the scale factor
            scale_factor = 1.0
            # open the catalogue
            f = h5py.File(catalogue, 'r')
            
            M_gas_temp = f['Subhalo/SubhaloMassType'][:,0]*1e10  #gas masses in Msun/h
            Met_gas_temp = f['Subhalo/SubhaloGasMetallicity'][:]  #gas metalicity
            M_star_temp = f['Subhalo/SubhaloMassType'][:,4]*1e10  #stellar masses in Msun/h
            Met_star_temp = f['Subhalo/SubhaloStarMetallicity'][:]  #stellar metalicity
            WDM_mass_temp = torch.ones(len(M_gas_temp))/context[num] #keV
            
            # close file
            f.close()
            
            temp = [M_star_temp,Met_star_temp,M_gas_temp,Met_gas_temp,WDM_mass_temp]
            
            non_zero_M_star = temp[0]>0
            
            
            for i in range(0,5):
                temp[i] = np.array(temp[i][non_zero_M_star])
                
                temp[i] = np.log10(temp[i]*10**10+1)-10
                temp[i][temp[i]==-10]=0
                temp[i] = torch.tensor(temp[i], dtype=torch.float32)
                self.data[i] = torch.cat((self.data[i], temp[i]),0)
                #print(temp[i].max(),temp[i].min())
            self.lense = len(self.data[0])
    
            
    def reshape(self):
        rand_perm = torch.randperm(self.lense)
        ori_data = torch.stack((self.data[0][rand_perm],self.data[1][rand_perm],self.data[2][rand_perm],self.data[3][rand_perm]),dim=0)
        context = self.data[4][rand_perm]

        self.data_mean = torch.std_mean(ori_data,dim=1)[1]
        self.data_std = torch.std_mean(ori_data,dim=1)[0]
        norm_data = self.normalizing(ori_data)

        context_mean = torch.std_mean(context)[1]
        context_std = torch.std_mean(context)[0]
        norm_context = (context-context_mean)/context_std

        self.train_num = int(0.8*self.lense)-int(0.8*self.lense)%self.BATCH_SIZE
        self.batch_num = int(self.train_num/self.BATCH_SIZE)
        train_indice = torch.linspace(0,self.train_num-1,self.train_num,dtype=int)
        test_indice = torch.linspace(int(0.8*self.lense)+1,self.lense-1,self.lense-int(0.8*self.lense)-1,dtype=int)

        self.train_source = torch.index_select(norm_data,1,train_indice)
        self.source_context = torch.index_select(norm_context,0,train_indice)

        self.test_data = torch.index_select(norm_data,1,test_indice)
        self.test_context = torch.index_select(norm_context,0,test_indice)

        self.train_data = torch.reshape(self.train_source,(4,self.batch_num,self.BATCH_SIZE))
        self.train_context = torch.reshape(self.source_context,(self.batch_num,self.BATCH_SIZE))
    
    def normalizing(self,samples):
        n = samples.size(0)
        result = torch.zeros(samples.size())
        for i in range(n):
            result[i] = (samples[i]-torch.std_mean(samples,dim=1)[1][i])/torch.std_mean(samples,dim=1)[0][i]
        return result
    
        
camels = CAMELS()

def define_model(trial):
    num_layers = trial.suggest_int("num_layers", 1, 10)
    input_dim = 4
    #num_layers = 1
    base_dist = dist.Normal(torch.zeros(input_dim).to(device), torch.ones(input_dim).to(device))

    

    transform = []
    #count_bins = 2**trial.suggest_int("count_bins", 4, 8)
    count_bins = 64
    bound = trial.suggest_int("bound", 2, 10)
    for i in range(0,num_layers):
        transform.append(T.conditional_spline_autoregressive(input_dim,context_dim=1, count_bins=64,bound=3).to(device))
    flow = dist.ConditionalTransformedDistribution(base_dist, transform)

    return flow,transform

def objective(trial):
    # Generate the model.
    flow,transform = define_model(trial)
    
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    weight_decay = trial.suggest_float("weight_decay", 1e-5, 0.1, log=True)

    optimizer = optim.Adam((par for model in transform for par in model.parameters()),lr = lr,weight_decay = weight_decay )

    camels.reshape()
    
    test = torch.t(camels.test_data)
    test = test.to(device)
    min_valid = 1e40
    fmodel = './CAMELS/models/WDM/WDM_con_%d.pt'%(trial.number)

    for i in range(10):
        rand_train = torch.randperm(camels.train_num)
        for j in range(0,4):
            camels.train_source[j] = camels.train_source[j][rand_train] 
        train_data = torch.reshape(camels.train_source,(4,camels.batch_num,camels.BATCH_SIZE))
        
        camels.source_context = camels.source_context[rand_train] 
        train_context = torch.reshape(camels.source_context,(camels.batch_num,camels.BATCH_SIZE,1))

        # batch loop:
        for num in range(camels.batch_num):
            x = torch.stack([train_data[j][num%(camels.batch_num)] for j in range(0,4)],dim=0)
            x = torch.t(x)
            x = x.to(device)
            
            y = train_context[num%(camels.batch_num)]
            y = y.to(device)
            
            optimizer.zero_grad()
            loss = -flow.condition(y.detach()).log_prob(x.detach()).mean()
            loss.backward()
            optimizer.step()
            flow.clear_cache()
        
        test = torch.t(camels.test_data).to(device)
        test_context = torch.reshape(camels.test_context,(len(camels.test_context),1))
        test_context = test_context.to(device)
        
        with torch.no_grad():
            error = -flow.condition(test_context).log_prob(test).mean()
        
        if error<min_valid:  
            min_valid = error
            torch.save(flow, fmodel)
        
        trial.report(min_valid,i)
        
        
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return min_valid


def main():
    #optuna.delete_study(study_name="CAMELS_WDM_con", storage="sqlite:///CAMELS_WDM_con.db")
    study = optuna.create_study(direction="minimize",study_name='CAMELS_WDM_con',storage='sqlite:///CAMELS_WDM_con.db',load_if_exists=True)
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(df)
    study.optimize(objective, n_trials=100,timeout = 3600*24)

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
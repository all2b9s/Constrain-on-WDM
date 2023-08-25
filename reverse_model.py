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
    BATCH_SIZE = 1024
    N_VARIABLE = 5
    N_CONTEXT = 2
    def __init__(self):
        data_0 = torch.tensor([])
        data_1 = torch.tensor([])
        data_2 = torch.tensor([])
        data_3 = torch.tensor([])
        data_4 = torch.tensor([])
        WDM_mass = torch.tensor([])
        Omega_m = torch.tensor([])

        wdm_ori =np.loadtxt('sobol_sequence_WDM_real_values.txt',usecols=[5])
        Omega_m_ori = np.loadtxt('sobol_sequence_WDM_real_values.txt',usecols=[0])

        self.data = [data_0,data_1,data_2,data_3,data_4,WDM_mass,Omega_m]
        for num in range(0,1024):
            
            
            # catalogue name
            catalogue = 'WDM/WDM_'+str(num)+'/fof_subhalo_tab_018.hdf5'

            # value of the scale factor
            scale_factor = 0.16
            # open the catalogue
            f = h5py.File(catalogue, 'r')
    
            data_0_temp = f['Subhalo/SubhaloStarMetallicity'][:]
            data_1_temp = f['Subhalo/SubhaloGasMetallicity'][:] 
            data_2_temp = f['Subhalo/SubhaloMassType'][:,0]*1e10
            data_3_temp = f['Subhalo/SubhaloMassType'][:,4]*1e10
            data_4_temp = f['Subhalo/SubhaloVmax'][:]
            
            WDM_mass_temp = torch.ones(len(data_0_temp))*wdm_ori[num]
            Omega_m_temp = torch.ones(len(data_0_temp))*Omega_m_ori[num]
            
            # close file
            f.close()
            
            temp = [data_0_temp,data_1_temp,data_2_temp,data_3_temp,data_4_temp,WDM_mass_temp,Omega_m_temp]
            
            non_zero_M_star = temp[0]>0
            
            
            non_zero_M_star = temp[3]>2e8
    
            #selected_indice = torch.linspace(0,1200-1,1200,dtype=int)
            
            for i in range(0,self.N_VARIABLE+self.N_CONTEXT):
                temp[i] = np.array(temp[i][non_zero_M_star])
                #temp[i] = np.array(temp[i])
                #temp[i] = temp[i][selected_indice]
                if i<self.N_VARIABLE:
                    temp[i] = np.log10(temp[i]*10**10+1)-10
                    temp[i][temp[i]==-10]=0
                temp[i] = torch.tensor(temp[i], dtype=torch.float32)
                self.data[i] = torch.cat((self.data[i], temp[i]),0)
                #print(temp[i].max(),temp[i].min())
    
            
    def reshape(self):
        lense = len(self.data[0])
            
        rand_perm = torch.randperm(lense)

        ori_data = torch.stack([self.data[i][rand_perm] for i in range(0,self.N_VARIABLE) ],dim=0)
        ori_context = torch.stack([self.data[i][rand_perm] for i in range(self.N_VARIABLE,self.N_VARIABLE+self.N_CONTEXT) ],dim=0)

        self.data_mean = torch.std_mean(ori_data,dim=1)[1]
        self.data_std = torch.std_mean(ori_data,dim=1)[0]
        norm_data = self.normalizing(ori_data)
        #print(norm_data)

        self.context_mean = torch.std_mean(ori_context,dim=1)[1]
        self.context_std = torch.std_mean(ori_context,dim=1)[0]
        norm_context= self.normalizing(ori_context)

        self.train_num = int(0.8*lense)-int(0.8*lense)%self.BATCH_SIZE
        self.batch_num = int(self.train_num/self.BATCH_SIZE)
        #print(train_num)
        train_indice = torch.linspace(0,self.train_num-1,self.train_num,dtype=int)
        test_indice = torch.linspace(int(0.8*lense)+1,lense-1,lense-int(0.8*lense)-1,dtype=int)

        self.source_data = torch.index_select(norm_data,1,train_indice)
        self.source_context = torch.index_select(norm_context,1,train_indice)


        self.train_data = torch.reshape(self.source_data,(self.N_VARIABLE,self.batch_num,self.BATCH_SIZE))
        self.test_data = torch.index_select(norm_data,1,test_indice)
        self.test_data = torch.t(self.test_data).to(device)

        self.train_context = torch.reshape(self.source_context,(self.N_CONTEXT,self.batch_num,self.BATCH_SIZE))
        self.test_context = torch.index_select(norm_context,1,test_indice)
        self.test_context = torch.t(self.test_context).to(device)
    
    def normalizing(self,samples):
        n = samples.size(0)
        result = torch.zeros(samples.size())
        for i in range(n):
            result[i] = (samples[i]-torch.std_mean(samples,dim=1)[1][i])/torch.std_mean(samples,dim=1)[0][i]
        return result
    
camels = CAMELS()

def define_model(trial):
    num_layers = trial.suggest_int("num_layers", 1, 5)
    input_dim = camels.N_CONTEXT
    context_dim = camels.N_VARIABLE  

    base_dist = dist.Normal(torch.zeros(input_dim).to(device), torch.ones(input_dim).to(device))


    transform = []
    count_bins = trial.suggest_int("count_bins", 2, 20)
    count_bins = 2
    bound = trial.suggest_int("bound", 2, 10)
    for i in range(0,num_layers):
        transform.append(
            T.conditional_spline_autoregressive(input_dim = input_dim, 
                                                context_dim=context_dim, 
                                                count_bins=count_bins,
                                                bound=bound).to(device)
            )
    flow = dist.ConditionalTransformedDistribution(base_dist, transform)

    return flow,transform

def objective(trial):
    # Generate the model.
    flow,transform = define_model(trial)
    
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 0.1, log=True)
    optimizer = optim.Adam((par for model in transform for par in model.parameters()),lr = lr,weight_decay = weight_decay )

    camels.reshape()
    
    min_valid = 1e40
    fmodel = './models/reverse/reverse_con_2_%d.pt'%(trial.number)
    
    num_iter = trial.suggest_int("num_iter", 1, 10)
    
    for i in range(0,num_iter):
        rand_train = torch.randperm(camels.train_num)
        for j in range(0,camels.N_VARIABLE):
            camels.source_data[j] = camels.source_data[j][rand_train] 
        train_data = torch.reshape(camels.source_data,(camels.N_VARIABLE,camels.batch_num,camels.BATCH_SIZE))
                
        for j in range(0,camels.N_CONTEXT):
            camels.source_context[j] = camels.source_context[j][rand_train] 
        train_context = torch.reshape(camels.source_context,(camels.N_CONTEXT,camels.batch_num,camels.BATCH_SIZE))

        # batch loop:
        for num in range(camels.batch_num):
            x = torch.stack([train_data[j][num%(camels.batch_num)] for j in range(0,camels.N_VARIABLE)],dim=0)
            x = torch.t(x)
            x = x.to(device)
            
            y = torch.stack([train_context[j][num%(camels.batch_num)] for j in range(0,camels.N_CONTEXT)],dim=0)
            y = torch.t(y)
            y = y.to(device)
            
            optimizer.zero_grad()
            loss = -flow.condition(x.detach()).log_prob(y.detach()).mean()
            loss.backward()
            optimizer.step()
            flow.clear_cache()
        
        with torch.no_grad():
            error = -flow.condition(camels.test_data).log_prob(camels.test_context).mean()
            #print(error)
            
        if error<min_valid:  
            min_valid = error
            torch.save(flow, fmodel)
        
        trial.report(min_valid,i)
        
        
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return min_valid

def main():
    #optuna.delete_study(study_name='CAMELS_reverse',storage='sqlite:///CAMELS_reverse.db')
    study = optuna.create_study(direction="minimize",study_name='CAMELS_reverse_2',storage='sqlite:///CAMELS_reverse_2.db',load_if_exists=True)
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
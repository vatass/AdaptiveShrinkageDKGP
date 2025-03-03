import pandas as pd
import numpy as np
from pandas.core.arrays.sparse import dtype
from pandas.core.indexes.base import default_index 
import sys
import torch
import gpytorch
from functions import *
import pickle
from exactgpmodels import SingleTaskDeepKernelNonLinear, SingleTaskDeepKernel
import argparse
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm import tqdm 
import wandb
import time 
import json
import math 
sns.set_style("white", {'axes.grid' : False})

# Plot Controls 
# sns.set_theme(context="paper",style="whitegrid", rc={'axes.grid' : True, 'font.serif': 'Times New Roman'})
# wandb.init(project="HMUSEDeepSingleTask", entity="vtassop", save_code=True)
parser = argparse.ArgumentParser(description='Temporal Deep Kernel Single Task GP model for a single HMUSE Roi')
## Data Parameters 
parser.add_argument("--gpuid", help="GPUs", default=0)
parser.add_argument("--experimentID", help="Indicates the Experiment Identifier", default='adniblsa') # 1adni normally
parser.add_argument("--exp", help="Indicates the modality", default='')
parser.add_argument("--kfoldID", help="Identifier for the Kfold IDs", default="missingdasae")
parser.add_argument("--file", help="Identifier for the data", default="subjectsamples_longclean_spare_adniblsa")
parser.add_argument("--covariates", help="String with the Covariates used as Condition in the Model", default="Diagnosis-Age-Sex-APOE4_Alleles-Education_Years")
parser.add_argument("--genomic", help="Indicates the genomic modality", default=0, type=int)
parser.add_argument("--followup", help="Indicates the followup information", default=0, type=int)
## Kernel Parameters ##
parser.add_argument('--kernel_choice', help='Indicates the selection of GP kernel', default='RBF', type=str)
parser.add_argument('--mean', help='Selection of Mean function', default='Constant', type=str)
## Deep Kernel Parameters 
parser.add_argument("--depth", help='indicates the depth of the Deep Kernel', default='')
parser.add_argument("--activ", help='indicates the activation function', default='relu')
parser.add_argument("--dropout", help='indicates the dropout rate', default=0.1, type=float)
## Training Parameters
parser.add_argument("--iterations", help="Epochs", default=100)
parser.add_argument("--optimizer", help='Optimizer', default='adam')
parser.add_argument("--learning_rate", help='Learning Rate', type=float, default=0.02)    # 0.01844 is in hmuse rois 
parser.add_argument("--task", help='Task id', type=str, default="SPARE_AD")  # Right Hippocampus 
parser.add_argument("--roi_idx", type=int, default=-1)
# Personalization # 
parser.add_argument("--personalization", type=str, default=False)
parser.add_argument("--history", type=int, default=4)
parser.add_argument("--pers_lr", type=float, help='Pers LR', default=0.01844) # 0.3510
parser.add_argument("--folder", type=int, default=2)

####
# Feature Representation 
# H_MUSE features
# Clinical_Features = Diagnosis, Age, Sex, APOE4 Alleles, Education Years
# Genomic 
# Follow-Up = {HMUSE}{Time}
# Time 
####

t0= time.time()
args = parser.parse_args()
history = int(args.history)
personalization = args.personalization
genomic = args.genomic 
gpu_id = int(args.gpuid) 
exp = args.exp 
iterations = int(args.iterations)
covariates = args.covariates.split("-")
kfoldID = args.kfoldID
file = args.file 
expID = args.experimentID 
genomic = args.genomic 
followup = args.followup
depth = args.depth 
activ = args.activ 
dropout = args.dropout
task = args.task 
kernel = args.kernel_choice
mean = args.mean
roi_idx = args.roi_idx

text_task = task 

personalization = False

mae_TempGP_list, mae_TempGP_list_comp = [], []  
population_results = {'id' : [], 'kfold': [], 'score': [], 'upper': [], 'lower': [], 'variance': [],  'y': [], 'time': []}
population_mae_kfold = {'kfold': [], 'mae': [], 'mse': [], 'rmse': [], 'R2': [],  'interval': [], 'coverage': []}
population_metrics_per_subject = {'kfold': [], 'id': [], 'mae_per_subject': [], 'wes_per_subject': [], 'interval': [], 'coverage': [] }

train_population_results = {'id':[], 'score': [], 'upper': [], 'lower': [], 'y': [], 'time': [], 'kfold': [] } 
train_population_mae_kfold = {'mae': [], 'kfold': [], 'interval': [], 'coverage': []}

if personalization: 
    personalization_results = {'id': [], 'kfold': [], 'score': [], 'upper': [], 'lower': [], 'variance': [],  'y': [], 'time': [], 'history_points': [], 'model': []}
    person_metrics_history = {'history': [], 'ae': [], 'se': [], 'coverage': [] ,  'model':[], 'interval': [], 'id': [], 'time': [], 'kfold': [] }
    person_mean_metrics = {'id': [], 'history': [], 'mae': [], 'mse': [], 'r2': [], 'rmse': [], 'model': [], 'kfold': [], 'coverage': [], 'interval': []}
    person_metrics_per_subject = {'kfold': [], 'id': [], 'mae_per_subject': [], 'wes_per_subject': [], 'history': [] }

# Configuration of W&B
# wandb.config = args
mae_MTGP_list, coverage_MTGP_list, interval_MTGP_list = [], [], [] 

folder = int(args.folder)
datasamples = pd.read_csv('/home/cbica/Desktop/LongGPClustering/data'+str(folder)+'/' + file + '.csv')
covariatesdf = pd.read_csv('/home/cbica/Desktop/LongGPClustering/data'+str(2)+'/covariates_subjectsamples_longclean_hmuse_adniblsa.csv')
longitudinal_covariates = pd.read_csv('/home/cbica/Desktop/LongGPClustering/data' + str(2) + '/longitudinal_covariates_subjectsamples_longclean_hmuse_convs_adniblsa.csv')
subject_ids = list(datasamples['PTID'].unique()) 

# wandb.config['Subjects'] = len(subject_ids) 

f = open('../LongGPClustering/roi_to_idx.json')
roi_to_idx = json.load(f)

for fold in range(5): 
    print('FOLD::', fold)
    train_ids, test_ids = [], []     

    # with (open("./data/train_subjects_cognitive_adni" + str(fold) +  ".pkl", "rb")) as openfile:
    with (open("/home/cbica/Desktop/LongGPClustering/data"+str(folder)+"/train_subject_adniblsa_ids_hmuse" + str(fold) +  ".pkl", "rb")) as openfile:
        while True:
            try:
                train_ids.append(pickle.load(openfile))
            except EOFError:
                break 
      
    # with (open("./data/test_subjects_cognitive_adni" + str(fold) +  ".pkl", "rb")) as openfile:
    with (open("/home/cbica/Desktop/LongGPClustering/data"+str(folder)+"/test_subject_adniblsa_ids_hmuse" + str(fold) +  ".pkl", "rb")) as openfile:

        while True:
            try:
                test_ids.append(pickle.load(openfile))
            except EOFError:
                break

    train_ids = train_ids[0]
    test_ids = test_ids[0]

    ### Use the first 200 subjects in order to extract the correct alphas
    # val_ids = train_ids[:200]

    print('Train IDs', len(train_ids))
    print('Test IDs', len(test_ids))
    print() 

    for t in test_ids: 
        if t in train_ids: 
            raise ValueError('Test Samples belong to the train!')


    ### SET UP THE TRAIN/TEST DATA FOR THE NON LINEAR DEEP KERNEL GP REGRESSION ### 
    train_x = datasamples[datasamples['PTID'].isin(train_ids)]['X']
    train_y = datasamples[datasamples['PTID'].isin(train_ids)]['Y']    
    test_x = datasamples[datasamples['PTID'].isin(test_ids)]['X']
    test_y = datasamples[datasamples['PTID'].isin(test_ids)]['Y']

    covariates_train = covariatesdf[covariatesdf['PTID'].isin(train_ids)]
    covariates_test = covariatesdf[covariatesdf['PTID'].isin(test_ids)]

    print('Covariates Test', covariates_test.shape)
    print('Covariates Train', covariates_train.shape)
    
    corresponding_test_ids = datasamples[datasamples['PTID'].isin(test_ids)]['PTID'].to_list()
    corresponding_train_ids = datasamples[datasamples['PTID'].isin(train_ids)]['PTID'].to_list() 
    assert len(corresponding_test_ids) == test_x.shape[0]
    assert len(corresponding_train_ids) == train_x.shape[0]

    train_x, train_y, test_x, test_y = process_temporal_singletask_data(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, test_ids=test_ids)
    # val_x, val_y, val_x, val_y = process_temporal_singletask_data(train_x=val_x, train_y=val_y, test_x=val_x, test_y=val_y, test_ids=test_ids)

    if torch.cuda.is_available():
        train_x = train_x.cuda(gpu_id) 
        train_y = train_y.cuda(gpu_id)#.squeeze()
        test_x = test_x.cuda(gpu_id) 
        test_y = test_y.cuda(gpu_id)#.squeeze() 
        # val_x = val_x.cuda(gpu_id)
        # val_y = val_y.cuda(gpu_id)

    ## Save the whole Data 
    torch.save(train_x, 'train_x_data_fold_' + str(fold) + '.pt')
    torch.save(train_x, 'train_y_data_fold_' + str(fold) + '.pt')

    print('Train Data', train_x.shape)
    print('Targets', train_y.shape)
    print('Test Data', test_x.shape)
    print('Targets', test_y.shape)

    if roi_idx != -1:
        print('Here', roi_idx)
        task = str(roi_idx)
        test_y = test_y[:, roi_idx]
        train_y = train_y[:, roi_idx]
    else: 
        print('To Infer the', task)
        roi_idx = 0 # It's either the inference of SPARE_AD or SPARE_BA 
        if task == 'SPARE_AD': 
            roi_idx = 0 
        else: 
            roi_idx=1 
        test_y = test_y[:, roi_idx]
        train_y = train_y[:, roi_idx]


    print(train_y.shape, test_y.shape)
    train_y = train_y.squeeze() 
    test_y = test_y.squeeze()

    #### DEFINE GP MODEL #### 
    # depth = [(train_x.shape[1], train_x.shape[1] * 2 ), (train_x.shape[1]*2, int(train_x.shape[1]/2))]
    depth = [(train_x.shape[1], int(train_x.shape[1]/2))]
    dr = args.dropout
    activ = 'relu'

    lat_dim = int(train_x.shape[1]/2)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    deepkernelmodel = SingleTaskDeepKernel(input_dim=train_x.shape[1], train_x=train_x, train_y=train_y, likelihood=likelihood, depth=depth, dropout=dr, activation=activ, kernel_choice=kernel, mean=mean,
     pretrained=False, feature_extractor=None, latent_dim=int(train_x.shape[1]/2), gphyper=None) 

    if torch.cuda.is_available(): 
        likelihood = likelihood.cuda(gpu_id) 
        deepkernelmodel = deepkernelmodel.cuda(gpu_id)

    training_iterations  =  iterations 
        
    # set up train mode 
    deepkernelmodel.feature_extractor.train()
    deepkernelmodel.train()
    deepkernelmodel.likelihood.train()

    if args.optimizer == 'adam': 
        optimizer = torch.optim.Adam([
        {'params': deepkernelmodel.feature_extractor.parameters(), 'lr': args.learning_rate},
        {'params': deepkernelmodel.covar_module.parameters(), 'lr': args.learning_rate },
        {'params': deepkernelmodel.mean_module.parameters(), 'lr': args.learning_rate},
        {'params': deepkernelmodel.likelihood.parameters(),  'lr': args.learning_rate} ], weight_decay=0.01) ## try more reg 
  

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, deepkernelmodel)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20, 50, 70], gamma=0.1)

    train_loss, val_loss = [], [] 
    for i in tqdm(range(training_iterations)):
        deepkernelmodel.train()
        likelihood.train()
        optimizer.zero_grad()
        output = deepkernelmodel(train_x)
        loss = -mll(output, train_y)
        loss.backward()

        # wandb.log({"mll_train_loss": loss})
        train_loss.append(loss.item())
        optimizer.step()

        deepkernelmodel.eval()
        likelihood.eval()

    # Set into eval mode
    deepkernelmodel.eval()
    likelihood.eval()

    # Make Predictions in Test 
    total_differences, total_derivatives = [], []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():

        f_preds = deepkernelmodel(test_x)
        y_preds = likelihood(f_preds)

        mean = y_preds.mean
        variance = y_preds.variance   ### What variance is this? Does it contain population variance?

        lower, upper = y_preds.confidence_region()

    mae_pop, ae_pop = mae(test_y.cpu().detach().numpy(), mean.cpu().detach().numpy())
    mse_pop, rmse_pop, se_pop = mse(test_y.cpu().detach().numpy(), mean.cpu().detach().numpy())  
    rsq = R2(test_y.cpu().detach().numpy(), mean.cpu().detach().numpy()) 

    # wandb.log({"test_mae": mae_pop})

    coverage, interval_width, mean_coverage, mean_interval_width  = calc_coverage(predictions=mean.cpu().detach().numpy(), groundtruth=test_y.cpu().detach().numpy(),
     intervals=[lower.cpu().detach().numpy(), upper.cpu().detach().numpy()])  

    coverage, interval_width, mean_coverage, mean_interval_width = coverage.numpy().astype(int), interval_width.numpy(), mean_coverage.numpy(), mean_interval_width.numpy() 

    # uncertainty eval
    acp = np.mean(coverage)
    ncp = acp - 0.5 

    # get the 50% width 
    posterior_std = y_preds.variance.sqrt()  # Standard deviation at each test point
    inverse_width = 1/(upper.cpu().detach().numpy()-lower.cpu().detach().numpy())

    mae_per_subject, wmae_per_subject, coverage_per_subject, interval_per_subject = calculate_errors(predictions=mean.cpu().detach().numpy(), upper_tensor=upper, lower_tensor=lower,    
                                ids=corresponding_test_ids, 
                                real_values=test_y.cpu().detach().numpy(),
                                real_tensor= test_y, 
                                weights=inverse_width.tolist())

    for key in mae_per_subject: 
        population_metrics_per_subject['kfold'].append(fold) 
        population_metrics_per_subject['id'].append(key) 
        population_metrics_per_subject['mae_per_subject'].append(mae_per_subject[key]) 
        population_metrics_per_subject['wes_per_subject'].append(wmae_per_subject[key])
        population_metrics_per_subject['interval'].append(coverage_per_subject[key])
        population_metrics_per_subject['coverage'].append(interval_per_subject[key])

    # wandb.log({"test_interval": mean_interval_width, "test_coverage": np.mean(coverage)})

    mae_MTGP_list.append(mae_pop)
    coverage_MTGP_list.append(np.mean(coverage))
    interval_MTGP_list.append(mean_interval_width)

    population_results['id'].extend(corresponding_test_ids) 
    population_results['kfold'].extend([fold for c in range(len(corresponding_test_ids))])
    population_results['score'].extend(mean.cpu().detach().numpy().tolist())
    population_results['lower'].extend(lower.cpu().detach().numpy().tolist())
    population_results['upper'].extend(upper.cpu().detach().numpy().tolist()) 
    population_results['variance'].extend(variance.cpu().detach().numpy().tolist()) 
    population_results['y'].extend(test_y.cpu().detach().numpy().tolist())
  
    time_ = test_x[:, -1].cpu().detach().numpy().tolist() 
    population_results['time'].extend(time_) 

    population_mae_kfold['mae'].append(mae_pop)
    population_mae_kfold['mse'].append(mse_pop)
    population_mae_kfold['rmse'].append(rmse_pop)
    population_mae_kfold['R2'].append(rsq)
    population_mae_kfold['kfold'].append(fold)
    population_mae_kfold['coverage'].append(np.mean(coverage))
    population_mae_kfold['interval'].append(mean_interval_width)

    ### Save the Population Hyperparams ###
    ## Extract the Population Model Parameters
    population_hyperparams = {} 
    ## Transfer the GP hyperparams 
    print('==== Population Hyperparams ====')
    for param_name, param in deepkernelmodel.named_parameters():
        if not param_name.startswith('feature'): 
            population_hyperparams[param_name] = param
        else: 
            print(param_name, param)
    print(population_hyperparams)

    ### Store the Model #####
    save_model(deepkernelmodel, optimizer, likelihood, filename='/home/cbica/Desktop/LongGPClustering/baselineonlyresults/population_deep_kernel_gp_'+ str(roi_idx) + '_' + str(fold)+'.pth')

    ### Extract the Feature Importance for Interpretability ###
    # Access the weights of the linear layer
    weights = deepkernelmodel.feature_extractor.final_linear.weight.cpu().detach()

    # Compute the feature importance as the absolute values of the weights
    feature_importance = torch.abs(weights).mean(dim=0)

    # Convert to numpy for easier handling, if needed
    feature_importance_np = feature_importance.numpy()

    # store 
    np.save('/home/cbica/Desktop/LongGPClustering/baselineonlyresults/feature_importance_' + str(task) + '_' + str(fold) + '.npy', feature_importance_np)

    personalization = False
    if personalization: 
        ## Keep the Deep Kernel Stable and train a new GP with the History of the Subject 
        print('START THE PERSONALIZATION!:: SS-DKGP')
        test_ids = list(set(corresponding_test_ids))
        print('Test IDs', test_ids)
        pers_mae, pers_interval, pers_coverage = [], [], [] 
        for test_subject in test_ids:  

            # Data # 
            test_x = datasamples[datasamples['PTID'].isin([test_subject])]['X']
            test_y = datasamples[datasamples['PTID'].isin([test_subject])]['Y']
           
            if test_x.shape[0] >= 7: 

                covariates_train = covariatesdf[covariatesdf['PTID'].isin([test_subject])]
                longitudinal_covariates_subject = longitudinal_covariates[longitudinal_covariates['PTID'].isin([test_subject])]

                test_x, test_y, _, _ = process_temporal_singletask_data(train_x=test_x, train_y=test_y, test_x=test_x, test_y=test_y, test_ids=test_ids)
                if torch.cuda.is_available():
                    test_x = test_x.cuda(gpu_id) 
                    test_y = test_y.cuda(gpu_id)#.squeeze() 

                test_y = test_y[:, roi_idx]
                test_y = test_y.squeeze()

                for h in [3,4,5,6]: 

                    # print('Test X, Y', test_x.shape, test_y.shape)
                    train_x_subject = test_x[:h, :]
                    train_y_subject = test_y[:h]

                    test_sub_x =  test_x[h:, :]
                    test_sub_y =  test_y[h:]

                    ## Do inference using the population model to extract the evaluation metrics ##
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        observed_pred = likelihood(deepkernelmodel(test_sub_x))   
                        preds = observed_pred.mean.cpu().detach().numpy()
                        lower, upper = observed_pred.confidence_region()
                        variance = observed_pred.variance.cpu().detach().numpy()

                    mae_pop, ae_pop = mae(test_sub_y.cpu().detach().numpy(), preds)
                    mse_pop, rmse_pop, se_pop = mse(test_sub_y.cpu().detach().numpy(), preds)  
                    rsq = R2(test_sub_y.cpu().detach().numpy(), preds) 
                    
                    coverage, interval_width, mean_coverage, mean_interval_width  = calc_coverage(predictions=preds, groundtruth=test_sub_y.cpu().detach().numpy(),
                    intervals=[lower.cpu().detach().numpy(), upper.cpu().detach().numpy()])  
                    coverage, interval_width, mean_coverage, mean_interval_width = coverage.numpy().astype(int), interval_width.numpy(), mean_coverage.numpy(), mean_interval_width.numpy() 
       
                    interval = np.abs(np.abs(upper.cpu().detach().numpy()) - np.abs(lower.cpu().detach().numpy()))
                    
                    time_ = test_sub_x[:, -1].cpu().detach().numpy().tolist()

                    person_metrics_history['id'].extend([test_subject for p in range(ae_pop.shape[0])])
                    person_metrics_history['time'].extend(test_sub_x[:,-1].tolist())
                    person_metrics_history['history'].extend([h for p in range(ae_pop.shape[0])])
                    person_metrics_history['ae'].extend(ae_pop.tolist())
                    person_metrics_history['se'].extend(se_pop.tolist())
                    person_metrics_history['model'].extend(['Population' for p in range(ae_pop.shape[0])])
                    person_metrics_history['interval'].extend(interval.tolist())
                    person_metrics_history['kfold'].extend([fold for p in range(ae_pop.shape[0])])
                    person_metrics_history['coverage'].extend(coverage.tolist())

                    person_mean_metrics['id'].append(test_subject)
                    person_mean_metrics['history'].append(h)
                    person_mean_metrics['mse'].append(mse_pop)
                    person_mean_metrics['mae'].append(mae_pop)
                    person_mean_metrics['rmse'].append(rmse_pop)
                    person_mean_metrics['r2'].append(rsq)
                    person_mean_metrics['model'].append('Population')
                    person_mean_metrics['kfold'].append(fold)
                    person_mean_metrics['coverage'].append(np.mean(coverage))
                    person_mean_metrics['interval'].append(np.mean(interval))

                    ## Do inference using the population model to visualize ##
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        observed_pred = likelihood(deepkernelmodel(test_x))   
                        preds = observed_pred.mean.cpu().detach().numpy()
                        lower, upper = observed_pred.confidence_region()
                        variance = observed_pred.variance.cpu().detach().numpy()
                    
                    time_ = test_x[:, -1].cpu().detach().numpy().tolist()
                    personalization_results['id'].extend([test_subject for k in range(len(time_))])
                    personalization_results['kfold'].extend([fold for k in range(len(time_))])
                    personalization_results['score'].extend(preds.tolist()) 
                    personalization_results['upper'].extend(upper.tolist())
                    personalization_results['lower'].extend(lower.tolist())
                    personalization_results['y'].extend(test_y.cpu().detach().numpy().tolist())
                    personalization_results['variance'].extend(variance.tolist())
                    personalization_results['time'].extend(time_)
                    personalization_results['history_points'].extend([h for k in range(len(time_))])
                    personalization_results['model'].extend(['Population' for k in range(len(time_))])

                    new_train = torch.cat((train_x, train_x_subject),0)
                    new_targets = torch.cat((train_y, train_y_subject), 0)

                    print('SS-DKGP Optimization')
                    print('New Data', new_train.shape, new_targets.shape)

                    perslikelihood = gpytorch.likelihoods.GaussianLikelihood()
                                      
                    persmodel =  SingleTaskDeepKernel(input_dim=train_x.shape[1],train_x=train_x_subject,train_y=train_y_subject,likelihood=perslikelihood, depth=depth,
                                        dropout=0.1, activation=activ, pretrained=True,latent_dim=lat_dim, feature_extractor=deepkernelmodel.feature_extractor,gphyper=population_hyperparams)
                   
                    mll = gpytorch.mlls.ExactMarginalLogLikelihood(perslikelihood, persmodel)

                    if torch.cuda.is_available(): 
                        perslikelihood = perslikelihood.cuda(gpu_id) 
                        persmodel = persmodel.cuda(gpu_id)

                    # set up train mode 
                    persmodel.train()
                    perslikelihood.train()

                    # subject-specific optimizer (ss-gp)
                    # 0.01844 
                    ss_optimizer = torch.optim.Adam([
                            {'params': persmodel.covar_module.parameters(), 'lr': args.learning_rate},
                            {'params': persmodel.mean_module.parameters(), 'lr': args.learning_rate},
                            {'params': persmodel.likelihood.parameters(),  'lr': args.learning_rate} ])

                    training_iterations_person = 500  ## it was 300 # it was 700

                    train_loss_per_subject = [] 
                    for q in range(training_iterations_person):
                        ss_optimizer.zero_grad()
                        output = persmodel(train_x_subject)
                        loss = -mll(output, train_y_subject)

                        loss.sum().backward()
                        # wandb.log({"pers_mll_train_loss_" + str(test_subject): loss.mean().item()})

                        train_loss_per_subject.append(loss.mean().item())
                        ss_optimizer.step()

                        if (q+1)%100 == 0 :
                            print('Iter %d/%d - Loss: %.3f' % (q + 1, training_iterations_person, loss.mean().item()))

                    # Eval 
                    # print('Evaluate!', test_x_subject.shape, test_y_subject.shape)
                    persmodel.eval()
                    perslikelihood.eval()
                    # calculate the performance on unseen data 
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        f_preds = persmodel(test_sub_x)
                        y_preds = perslikelihood(f_preds)
                        mean_1 = y_preds.mean
                        lower_1, upper_1 = y_preds.confidence_region()

                    coverage, interval_width, mean_coverage, mean_interval_width  = calc_coverage(predictions=mean_1.cpu().detach().numpy(), groundtruth=test_sub_y.cpu().detach().numpy(),
                    intervals=[lower_1.cpu().detach().numpy(), upper_1.cpu().detach().numpy()])  

                    coverage, interval_width, mean_coverage, mean_interval_width = coverage.numpy().astype(int), interval_width.numpy(), mean_coverage.numpy(), mean_interval_width.numpy() 
                    
                    mae_pers, ae_pers = mae(test_sub_y.cpu().detach().numpy(), mean_1.cpu().detach().numpy())
                    mse_pers, rmse_pers, se_pers = mse(test_sub_y.cpu().detach().numpy(), mean_1.cpu().detach().numpy())  
                    rsq = R2(test_sub_y.cpu().detach().numpy(), mean_1.cpu().detach().numpy()) 
                    
                    coverage, interval_width, mean_coverage, mean_interval_width  = calc_coverage(predictions= mean_1.cpu().detach().numpy(), groundtruth=test_sub_y.cpu().detach().numpy(),
                    intervals=[lower_1.cpu().detach().numpy(), upper_1.cpu().detach().numpy()])  
                    coverage, interval_width, mean_coverage, mean_interval_width = coverage.numpy().astype(int), interval_width.numpy(), mean_coverage.numpy(), mean_interval_width.numpy() 

                    interval = np.abs(np.abs(upper_1.cpu().detach().numpy()) - np.abs(lower_1.cpu().detach().numpy()))

                    # uncertainty eval
                    acp = np.mean(coverage)
                    ncp = acp - 0.5 

                    # get the 50% width 
                    posterior_std = y_preds.variance.sqrt()  # Standard deviation at each test point
                    inverse_width = 1/(upper_1.cpu().detach().numpy()-lower_1.cpu().detach().numpy())

                    mae_per_subject, wmae_per_subject, coverage_per_subject, interval_per_subject = calculate_errors(predictions=mean_1.cpu().detach().numpy(), upper_tensor=upper_1, lower_tensor=lower_1,    
                                ids=corresponding_test_ids, 
                                real_values=test_sub_y.cpu().detach().numpy(),
                                real_tensor= test_sub_y, 
                                weights=inverse_width.tolist())

                    for key in mae_per_subject: 
                        person_metrics_per_subject['kfold'].append(fold) 
                        person_metrics_per_subject['id'].append(key) 
                        person_metrics_per_subject['mae_per_subject'].append(mae_per_subject[key]) 
                        person_metrics_per_subject['wes_per_subject'].append(wmae_per_subject[key])
                        person_metrics_per_subject['history'].append(h)


                    interval = np.abs(np.abs(upper_1.cpu().detach().numpy()) - np.abs(lower_1.cpu().detach().numpy()))
                    person_metrics_history['time'].extend(test_sub_x[:,-1].tolist())
                    person_metrics_history['id'].extend([test_subject for p in range(ae_pers.shape[0])])
                    person_metrics_history['history'].extend([h for p in range(ae_pers.shape[0])])
                    person_metrics_history['ae'].extend(ae_pers.tolist())
                    person_metrics_history['se'].extend(se_pers.tolist())
                    person_metrics_history['model'].extend(['Personalized' for p in range(ae_pers.shape[0])])
                    person_metrics_history['interval'].extend(interval.tolist())
                    person_metrics_history['kfold'].extend([fold for p in range(ae_pers.shape[0])])
                    person_metrics_history['coverage'].extend(coverage.tolist())

                    person_mean_metrics['id'].append(test_subject)
                    person_mean_metrics['history'].append(h)
                    person_mean_metrics['mse'].append(mse_pers)
                    person_mean_metrics['mae'].append(mae_pers)
                    person_mean_metrics['rmse'].append(rmse_pers)
                    person_mean_metrics['r2'].append(rsq)
                    person_mean_metrics['model'].append('Personalized')
                    person_mean_metrics['kfold'].append(fold)
                    person_mean_metrics['coverage'].append(np.mean(coverage))
                    person_mean_metrics['interval'].append(np.mean(interval))

                    # visualize the predictions in all subject data
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        f_preds = persmodel(test_x)
                        y_preds = perslikelihood(f_preds)
                        mean_11 = y_preds.mean
                        lower_11, upper_11 = y_preds.confidence_region()
                        variance = y_preds.variance

                    time_ = test_x[:, -1].cpu().detach().numpy().tolist() 

                    personalization_results['id'].extend([test_subject for k in range(len(time_))])
                    personalization_results['kfold'].extend([fold for k in range(len(time_))])
                    personalization_results['score'].extend(mean_11.cpu().detach().numpy().tolist()) 
                    personalization_results['upper'].extend(lower_11.cpu().detach().numpy().tolist())
                    personalization_results['lower'].extend(upper_11.cpu().detach().numpy().tolist())
                    personalization_results['variance'].extend(variance.cpu().detach().numpy().tolist())
                    personalization_results['y'].extend(test_y.cpu().detach().numpy().tolist())
                    personalization_results['time'].extend(time_)
                    personalization_results['history_points'].extend([h for k in range(len(time_))])
                    personalization_results['model'].extend(['SS-DKGP' for k in range(len(time_))])
    # del deepkernelmodel
    # del likelihood
    del train_x
    del train_y 
    del test_x
    del test_y 
    torch.cuda.empty_cache() 

if personalization:

    # print('id', len(personalization_results['id']))
    # print('kfold', len(personalization_results['kfold']))
    # print('score', len(personalization_results['score']))
    # print('upper', len(personalization_results['upper']))
    # print('lower', len(personalization_results['lower']))
    # print('variance', len(personalization_results['variance']))
    # print('time', len(personalization_results['time']))
    # print('history points', len(personalization_results['history_points']))
    # print('model', len(personalization_results['model']))

    # print('id', len(person_metrics_history['id']))
    # print('time', len(person_metrics_history['time']))
    # print('history', len(person_metrics_history['history']))
    # print('ae', len(person_metrics_history['ae']))
    # print('se', len(person_metrics_history['se']))
    # print('model', len(person_metrics_history['model']))
    # print('interval', len(person_metrics_history['interval']))
    # print('kfold', len(person_metrics_history['kfold']))
    # print('coverage', len(person_metrics_history['coverage']))

    # print('id', len(person_mean_metrics['id']))
    # print('history', len(person_mean_metrics['history']))
    # print('mae', len(person_mean_metrics['mae']))
    # print('mse', len(person_mean_metrics['mse']))
    # print('r2', len(person_mean_metrics['r2']))
    # print('rmse', len(person_mean_metrics['rmse']))
    # print('model', len(person_mean_metrics['model']))
    # print('kfold', len(person_mean_metrics['kfold']))
    # print('coverage', len(person_mean_metrics['coverage']))
    # print('interval', len(person_mean_metrics['interval']))


    personalization_results_df = pd.DataFrame(data=personalization_results)
    personalization_results_df.to_csv('./neuripsresults/person_ss_singletask_' + str(task) + '_dkgp_population_'+ expID+'.csv')

    person_metrics_history_df = pd.DataFrame(data=person_metrics_history)
    person_mean_metrics_df = pd.DataFrame(data=person_mean_metrics) 
    person_metrics_per_subject_df = pd.DataFrame(data=person_metrics_per_subject)

    person_metrics_history_df.to_csv('./neuripsresults/person_ss_metrics_results_dkgp_'+ str(task) + '_' + expID +'.csv')
    person_mean_metrics_df.to_csv('./neuripsresults/person_ss_mean_metrics_dkgp_' + str(task) + '_' + expID +'.csv')
    person_metrics_per_subject_df.to_csv('./neuripsresults/person_ss_metrics_persubject_dkgp_' + str(task) + '_' + expID +'.csv')

print('#### Evaluation of the Singletask Deep Kernel Temporal GP for ROI ' + str(text_task) + '###')
population_mae_kfold_df = pd.DataFrame(data=population_mae_kfold)
population_mae_kfold_df.to_csv('./neuripsresults/singletask_'+ str(text_task) + '_dkgp_mae_kfold_'+ expID+'.csv')

population_results_df = pd.DataFrame(data=population_results)
population_results_df.to_csv('./neuripsresults/singletask_' + str(text_task) + '_dkgp_population_'+ expID+'.csv')

population_metrics_per_subject_df = pd.DataFrame(data=population_metrics_per_subject)
population_metrics_per_subject_df.to_csv('./neuripsresults/singletask_' + str(text_task) + '_dkgp_mae_per_subject_kfold_' + expID + '.csv')

print('#### Evaluation of the Singletask Deep Kernel Temporal GP for ROI ' + str(text_task) + '###')
print('POPULATION GP')
print('mean:', np.mean(mae_MTGP_list, axis=0))
print('var:', np.var(mae_MTGP_list, axis=0))
print('Interval', np.mean(interval_MTGP_list), np.var(interval_MTGP_list))
print('Coverage', np.mean(coverage_MTGP_list), np.var(coverage_MTGP_list))
# wandb.log({'mean MAE': np.mean(np.mean(mae_MTGP_list, axis=0)), "mean Interval": np.mean(np.mean(interval_MTGP_list, axis=0)), "mean Coverage": np.mean(np.mean(coverage_MTGP_list, axis=0))})

if personalization: 
    print("### Evaluation of the Personalization ###")
    print('mean:', np.mean(person_mean_metrics['mae'], axis=0))
    print('var:', np.var(person_mean_metrics['mae'], axis=0))
    print('Interval', np.mean(person_metrics_history['interval']), np.var(person_metrics_history['interval']))
    print('Coverage', np.mean(person_metrics_history['coverage']), np.var(person_metrics_history['coverage']))
    # wandb.log({'Person mean MAE':np.mean(person_mean_metrics['mae'], axis=0), "Person mean Interval": np.mean(person_metrics_history['interval'], axis=0), "Person mean Coverage": np.mean(person_metrics_history['coverage'], axis=0)})

t1 = time.time() - t0 
print("Time elapsed: ", t1) 

"""
Subject-Specific Deep Kernel Gaussian Process model.
"""
from typing import Dict, Optional, Tuple, Union
import torch
import gpytorch
from .base import BaseDeepKernel

class SubjectSpecificDKGP(BaseDeepKernel):
    """Subject-Specific Deep Kernel Gaussian Process model.
    
    This model is initialized with parameters from a population model
    and fine-tuned on subject-specific data.
    """
    
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        input_dim: int,
        latent_dim: int,
        population_params: Optional[Dict[str, torch.Tensor]] = None,
        depth: int = 2,
        dropout: float = 0.1,
        activation: str = 'relu',
        kernel: str = 'RBF',
        mean: str = 'constant',
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        """Initialize the subject-specific DKGP model.
        
        Args:
            train_x: Training input data
            train_y: Training target data
            input_dim: Dimension of input features
            latent_dim: Dimension of latent space
            population_params: Optional parameters from population model
            depth: Number of layers in feature extractor
            dropout: Dropout rate
            activation: Activation function ('relu', 'tanh', etc.)
            kernel: Kernel function ('RBF' or 'Matern')
            mean: Mean function ('constant' or 'linear')
            learning_rate: Learning rate for optimization
            n_epochs: Number of training epochs
            device: Device to run the model on
        """
        # Initialize likelihood
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        super().__init__(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            input_dim=input_dim,
            latent_dim=latent_dim,
            depth=depth,
            dropout=dropout,
            activation=activation,
            kernel=kernel,
            mean=mean
        )
        
        # Load population parameters if provided
        if population_params is not None:
            self.feature_extractor.load_state_dict(population_params)
            # Freeze feature extractor parameters
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.device = device
        self.to(device)
    
    def fit(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        val_x: Optional[torch.Tensor] = None,
        val_y: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> Dict[str, list]:
        """Train the subject-specific DKGP model.
        
        Args:
            train_x: Training input data
            train_y: Training target data
            val_x: Optional validation input data
            val_y: Optional validation target data
            verbose: Whether to print training progress
            
        Returns:
            Dictionary containing training history
        """
        self.train()
        # Only optimize GP parameters, feature extractor is frozen
        optimizer = torch.optim.Adam([
            {'params': self.covar_module.parameters(), 'lr': self.learning_rate},
            {'params': self.mean_module.parameters(), 'lr': self.learning_rate},
            {'params': self.likelihood.parameters(), 'lr': self.learning_rate}
        ])
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        
        history = {
            'train_loss': [],
            'val_loss': [] if val_x is not None else None
        }
        
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            output = self(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            
            history['train_loss'].append(loss.item())
            
            if val_x is not None:
                with torch.no_grad():
                    val_output = self(val_x)
                    val_loss = -mll(val_output, val_y)
                    history['val_loss'].append(val_loss.item())
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f'Epoch {epoch+1}/{self.n_epochs} - Loss: {loss.item():.4f}')
        
        return history
    
    def unfreeze_feature_extractor(self) -> None:
        """Unfreeze the feature extractor parameters for fine-tuning."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
    
    def save_model(self, path: str) -> None:
        """Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'likelihood_state_dict': self.likelihood.state_dict(),
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim
        }, path)
    
    @classmethod
    def load_model(
        cls,
        path: str,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> 'SubjectSpecificDKGP':
        """Load a saved model from disk.
        
        Args:
            path: Path to the saved model
            train_x: Training input data
            train_y: Training target data
            device: Device to load the model on
            
        Returns:
            Loaded SubjectSpecificDKGP model
        """
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            train_x=train_x,
            train_y=train_y,
            input_dim=checkpoint['input_dim'],
            latent_dim=checkpoint['latent_dim'],
            device=device
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        
        return model




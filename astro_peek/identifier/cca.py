import numpy as np 
from cca_zoo.linear import CCA
from cca_zoo.linear import MCCA

class CCA_Score:
    def __init__(self, model_list):
        self.model_list=model_list # should have shape (nmodels, nsamples, ndims) where ndims is the number of latent dimensions
        self.ndims=self.model_list.shape[2]
        self.nsamples = self.model_list.shape[1]
        
    def compute_all_pairwise_cca(self, train_frac=0.8):
        # models must have the same latent dimensions, otherwise PCA must be applied first 
        # split the samples into train and test data for training the CCA model and then calculating the scores 
        ntrain = train_frac * self.nsamples 
        train_views = self.model_list[:, :ntrain, :]
        test_views = self.model_list[:, ntrain:, :]
        
        # initialize the MCCA model and train 
        model_mcca = MCCA(latent_dimensions=self.ndims).fit(train_views) 
        
        # calculate the pairwise CCA values for each pair of models
        corrs_MCCA = model_mcca.pairwise_correlations(test_views)
        
        return corrs_MCCA
    
    def compute_one_pair_cca(self, train_frac=0.8):
        # used if model_list has only two models 
        ntrain = train_frac * self.nsamples 
        train_views = self.model_list[:, :ntrain, :]
        test_views = self.model_list[:, ntrain:, :]
        
        model = CCA(latent_dimensions=self.ndims).fit(train_views) 
        
        corrs = model.score(test_views)
        
        return corrs
        
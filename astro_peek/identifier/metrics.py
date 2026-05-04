import numpy as np 
from cca_zoo.linear import CCA
from cca_zoo.linear import MCCA
import matplotlib.pyplot as plt
from itertools import combinations 
from torch import nn
from tqdm import tqdm 

class CCA_Score:
    def __init__(self, model_list):
        self.model_list=model_list # should have shape (nmodels, nsamples, ndims) where ndims is the number of latent dimensions. Data should be already mapped to latents before applying CCA
        self.ndims=self.model_list.shape[-1]
        self.num_models = self.model_list.shape[0]
        
    def compute_all_pairwise_cca(self):
        '''This function computes the pairwise CCA values of all possible pairs of models, using the self.model_list. It requires that all models have the same latent dimensions and 
        number of samples. 
        parameters: 
        - self.model_list: shape of (nmodels, nsamples, nlatents)
        
        returns: 
        - corrs_MCCA: a list of correlated pairwise CCA values for each possible pair of models. Shape of (nmodels, nmodels, nlatents)
        
        NOTE: to access the correlated CCA value of model N and model M for latent R, look at corrs_MCCA[N-1,M-1,R-1]
        '''
        
        # fit the CCA model with the latents
        views = self.model_list
        
        # initialize the MCCA model and train 
        model_mcca = MCCA(latent_dimensions=self.ndims).fit(views) 
        
        # calculate the pairwise CCA values for each pair of models
        corrs_MCCA = model_mcca.pairwise_correlations(views)
        
        return corrs_MCCA
    
    def compute_one_pair_cca(self):
        '''This function computes the pairwise CCA values for all latent dimensions between two models only. Use only if your model_list contains just two models. 
        parameters: 
        - self.model_list: shape of (2, nsamples, nlatents)
        
        returns: 
        - corrs: a list of correlated pairwise CCA values between the two models for each latent. Shape of (nlatents,)'''
        views = self.model_list
        
        model = CCA(latent_dimensions=self.ndims).fit(views) 
        
        corrs = model.score(views)
        
        return corrs
    
    def calculate_mean_cca(self, corrs):
        '''This function computes the mean CCA values for each latent across all models. 
        parameters: 
        - corrs: shape of (nmodels, nmodels, nlatents)
        
        returns: 
        - mean_corrs: mean CCA values across all models for each latent. Shape of (nlatents,)'''
        
        # get the upper triangle of the matrix and exclude the diagonal 
        iu = np.triu_indices(corrs.shape[0], k=1)

        # means shape: (nlatents,)
        mean_corrs = corrs[iu[0], iu[1], :].mean(axis=0)
        
        return mean_corrs
    
    def plot_cca_v_latent(self, corrs, save_loc=None, models=[0,1], encoder_type='f(x)', mean=False):
        '''This function plots the pairwise CCA values of two models for all latents.
        parameters: 
        - corrs: the calculated CCA values. Shape of either (nmodels, nmodels, nlatents) if you have more than 2 models, else shape (nlatents,)
        - save_loc: directory where you want to save the plot. Leave as None if you don't want to save it
        - models: the models you want to compare (numbered according to python indexing)'''
        
        if corrs.ndim == 1:
            i, j = 0,1
            scores = corrs
            if mean:
                title = "Mean CCA value across all models vs latent dimension for "+encoder_type
                ylabel = "Mean CCA value"
            else: 
                title = "CCA value vs latent dimenion for "+encoder_type+", models "+str(i+1)+" and "+str(j+1)
                ylabel = "CCA value"
            
        else:
            i, j = models[0], models[1] 
            scores = corrs[i,j,:]
            title = "CCA value vs latent dimenion for "+encoder_type+", models "+str(i+1)+" and "+str(j+1)
            ylabel = "CCA value"
        
        dims = np.arange(1, self.ndims+1)
        
        plt.figure(figsize=(13,4))
        plt.plot(dims, scores, "o", label="CCA")
        plt.xlabel("Latent dimension")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(dims)
        plt.ylim(0,1)
        if save_loc is not None: 
            plt.savefig(save_loc+'CCA_v_latents_'+encoder_type+f'_models_{i}_and_{j}.png')
        plt.show()
        


class R2_score: 
    def __init__(self, latent_list): 
        """
        Wrapper class to compute a R2 score in a list of latent vectors.  
        latent_list = list of latent vectors produced by different models. Shape: (nmodels, nsamples, ndims) 
        """
    #TODO
        self.ndims = latent_list.shape[-1]
        self.latent_list = latent_list

    def _fit_matrix(self, latent_a, latent_b, num_iter = 1000, lr = 1e-3):
        linear_model = nn.Linear(self.ndims, self.ndims) 
        optimizer = nn.Adam(linear_model.parameters(), lr = lr)
        loss_fn = nn.MSELoss(reduction = "mean")

        for i in tqdm(range(num_iter)): 
            optimizer.zero_grad()
            predicted_latent_b = linear_model(latent_a)
            loss = loss_fn(latent_b, predicted_latent_b)
            loss.backward()
            optimizer.step()

        return linear_model, loss

    def compute_pair_score(self, latent_a, latent_b, fitter_args = {}): 
        """
        Computes the R2 score for a pair of self.latent_list produced by two models M_1 and M_2
        """
        ...
        self._fit_matrix(latent_a, latent_b, **fitter_args)

    def compute_score(self, fitter_args = {}): 
        """
        Computes the R2 scores between every possible pair of self.latent_list 

        Output: (len(latent_list), ) = R2 scores.
        """
        # Computing every possible pairs 
        possible_pairs_idx = combinations(np.arange(len(self.latent_list)), r = 2) # just 2 pairs of models at a time
        scores = np.empty(shape = (len(possible_pairs_idx),), dtype = np.float32)
        
        for i, possible_pair in enumerate(possible_pairs_idx): 
            latent_1 = self.latent_list[possible_pair[0]]
            latent_2 = self.latent_list[possible_pair[1]]
            scores[i] = self.compute_pair_score(latent_1, latent_2, **fitter_args)

        return scores


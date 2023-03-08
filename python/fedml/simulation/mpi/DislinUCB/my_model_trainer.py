import torch
from torch import nn
import numpy as np
import copy

from ...core.alg_frame.client_trainer import ClientTrainer
import logging
from util_functions import featureUniform, gaussianFeature, dsigmoid, sigmoid

class MyModelTrainer(ClientTrainer):

    def __init__(self, model, args):
       super().__init__(model, args)
       self.cpu_transfer = False if not hasattr(self.args, "cpu_transfer") else self.args.cpu_transfer


    def train(self, A_local, b_local, A_uploadbuffer, b_uploadbuffer, numObs_local, AInv, UserTheta,
                alpha_t, d, articlepool, device, args):

        maxPTA = float('-inf')
        articlePicked = None
        numerator = np.linalg.det(A_local+self.lambda_ * np.identity(n=self.d))
        denominator = np.linalg.det(A_local-A_uploadbuffer+self.lambda_ * np.identity(n=self.d))

        while(np.log(numerator/denominator)*(self.numObs_uploadbuffer) < self.threshold):
            for x in articlepool: 
                x_pta = self.getUCB(alpha_t, x.featureVector)
                # pick article with highest UCB score
                # Going to need an arm class
                if maxPTA < x_pta:
                    articlePicked = x
                    maxPTA = x_pta
            
            inner_prod = np.dot(self.theta, articlePicked.featureVector)

            if self.reward_model == 'linear':
                reward = inner_prod
            elif self.reward_model == 'sigmoid':
                reward = sigmoid(inner_prod)
            else:
                raise ValueError
            
            A_local += np.outer(articlePicked.featureVector, articlePicked.featureVector)
            b_local += articlePicked.FeatureVector * reward
            self.numObs_local += 1

            A_uploadbuffer += np.outer(articlePicked.featureVector, articlePicked.featureVector)
            b_uploadbuffer += articlePicked.featureVector * reward
            numObs_uploadbuffer += 1

            AInv = np.linalg.inv(self.A_local+self.lambda_ * np.identity(n=self.d))
            UserTheta = np.dot(self.AInv, self.b_local)

            self.alpha_t = self.NoiseScale * np.sqrt(
                self.d * np.log(1 + (self.numObs_local) / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_)) + np.sqrt(
                self.lambda_)
            
            numerator = np.linalg.det(self.A_local+self.lambda_ * np.identity(n=self.d))
            denominator = np.linalg.det(self.A_local-self.A_uploadbuffer+self.lambda_ * np.identity(n=self.d))

        return A_uploadbuffer, b_uploadbuffer


    def getUCB(self, alpha, article_FeatureVector):
        if alpha == -1:
            alpha = self.alpha_t

        mean = np.dot(self.UserTheta, article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv), article_FeatureVector))
        pta = mean + alpha * var
        return pta
    

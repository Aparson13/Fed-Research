import time

from fedml.data import split_data_for_dist_trainers
from ...constants import FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL
from ...core.mlops.mlops_profiler_event import MLOpsProfilerEvent
from ..util_functions import featureUniform, gaussianFeature, dsigmoid, sigmoid
import numpy as np
from .ArticleManager import ArticleManager
import random
from datetime import datetime
from random import sample, shuffle


class FedMLTrainer(object):
    def __init__(
        self,
        args,
        client_index,
        device,
        dim,
        lambda_,
        alpha,
        delta_,
        threshold,
        n_articles,
    ):
        
        self.d = dim
        self.lambda_ = lambda_
        # self.delta_ = delta_
        self.delta_ = 1e-1
        self.alpha = alpha
        self.device = device
        self.client_index = client_index
        self.threshold = threshold
        self.n_articles = n_articles
        self.A_local = np.zeros((self.d, self.d))
        self.b_local = np.zeros(self.d)
        self.numObs_local = 0
        self.A_uploadbuffer = np.zeros((self.d, self.d))
        self.b_uploadbuffer = np.zeros(self.d)
        self.numObs_uploadbuffer = 0
        self.poolArticleSize = 25
        # for computing UCB
        self.AInv = np.linalg.inv(self.A_local+self.lambda_ * np.identity(n=self.d))
        self.NoiseScale = 0.1
        self.UserTheta = np.zeros(self.d)
        self.alpha_t = self.NoiseScale * (np.sqrt(self.d * np.log(1 + self.numObs_local)/ (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_) + np.sqrt(self.lambda_))
        self.reward_model = "linear"
        self.sync = False
        AM = ArticleManager(dim, n_articles, 0, gaussianFeature, argv={'l2_limit': 1})
        self.AM = AM.simulateArticlePool()
        self.regulateArticlePool()

    def train(self, args):

        maxPTA = float('-inf')
        random.seed(self.client_index)
        articlePicked = None
        numerator = np.linalg.det(self.A_local+self.lambda_ * np.identity(n=self.d))
        denominator = np.linalg.det(self.A_local-self.A_uploadbuffer+self.lambda_ * np.identity(n=self.d))
        training_round = 0
        
        while(np.log(numerator/denominator)*(self.numObs_uploadbuffer) < (self.threshold * (self.client_index+1)**2) and self.sync == False):
            
            for x in self.articlePool: 
                x_pta = self.getUCB(self.alpha_t, x.featureVector)
                x_pta = x_pta 
                # pick article with highest UCB score *  random.random()
                # Going to need an arm class
                if maxPTA < x_pta:
                    articlePicked = x
                    maxPTA = x_pta
            
            inner_prod = np.dot(self.UserTheta, articlePicked.featureVector)

            if self.reward_model == 'linear':
                reward = inner_prod
            elif self.reward_model == 'sigmoid':
                reward = sigmoid(inner_prod)
            else:
                raise ValueError
            
            # articlePicked.featureVector = articlePicked.featureVector * random.random()
            
            self.A_local += np.outer(articlePicked.featureVector, articlePicked.featureVector)
            self.b_local += articlePicked.featureVector * reward
            self.numObs_local += 1

            self.A_uploadbuffer += np.outer(articlePicked.featureVector, articlePicked.featureVector)
            self.b_uploadbuffer += articlePicked.featureVector * reward
            self.numObs_uploadbuffer += 1

            self.AInv = np.linalg.inv(self.A_local+self.lambda_ * np.identity(n=self.d))
            self.UserTheta = np.dot(self.AInv, self.b_local)

            self.alpha_t = self.NoiseScale * np.sqrt(
                self.d * np.log(1 + (self.numObs_local) / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_)) + np.sqrt(
                self.lambda_)
            
            
            numerator = np.linalg.det(self.A_local+self.lambda_ * np.identity(n=self.d))
            denominator = np.linalg.det(self.A_local-self.A_uploadbuffer+self.lambda_ * np.identity(n=self.d))
            training_round = training_round  + 1

            # print("-"*50 + "\n"*10 + "Client index: " + str(self.client_index) + "\n"*2
            #        +  "MaxPTA:" + str(maxPTA) + "\n"*2
            #     #    +  "det(V):" + str(np.log(numerator/denominator)*(self.numObs_uploadbuffer)) + "\n"*2
            #     #    + "Threshold: " + str(self.threshold) + "\n"*2 
            #     #    + "Sync Signal: " + str(self.sync) + "\n"*2 
            #     #    + "Training Round: " + str(training_round) + "\n"*2 
            #     #    + "Time: " + str(datetime.now()) + "\n"*2 
            #        + "\n"*10 + "-" *50)
        print("-"*50 + "\n"*10 + "Client index: " + str(self.client_index) + "\n"*2
                   +  "MaxPTA:" + str(maxPTA) + "\n"*2
                #    +  "det(V):" + str(np.log(numerator/denominator)*(self.numObs_uploadbuffer)) + "\n"*2
                #    + "Threshold: " + str(self.threshold) + "\n"*2 
                #    + "Sync Signal: " + str(self.sync) + "\n"*2 
                #    + "Training Round: " + str(training_round) + "\n"*2 
                #    + "Time: " + str(datetime.now()) + "\n"*2 
                   + "\n"*10 + "-" *50)
        print("-"*50 + "\n"*10 + "Training Finished after " + str(training_round) + " rounds."  "\n"*2 + "Sync signal: " + str(self.sync)+ "\n"*10 + "-"*50)
        return self.A_uploadbuffer, self.b_uploadbuffer


    def getUCB(self, alpha, article_FeatureVector):
        if self.alpha == -1:
            alpha = self.alpha_t

        mean = np.dot(self.UserTheta, article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv), article_FeatureVector))
        pta = mean + alpha * var
        return pta
    

    def update_dataset(self, global_model_params, client_index):
        print("-"*50 + "\n"*10 + "Updating the Dataset. " + "\n"*10 + "-"*50)
        self.A_local = global_model_params[0]
        self.b_local = global_model_params[1]
        # self.A_local += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        # self.b_local += articlePicked_FeatureVector * click
        self.numObs_local += 1

        self.A_uploadbuffer = np.zeros((self.d, self.d))
        self.b_uploadbuffer = np.zeros(self.d)
        self.numObs_uploadbuffer = 0

        # self.AInv = np.linalg.inv(self.A_local+self.lambda_ * np.identity(n=self.d))
        # self.UserTheta = np.dot(self.AInv, self.b_local)

        # self.alpha_t = np.sqrt(
        # self.d * np.log(1 + (self.numObs_local) / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_)) + np.sqrt(
        # self.lambda_)
        self.sync = False

    def getTheta(self):
        return self.UserTheta
    
    def regulateArticlePool(self):
        # Randomly generate articles
        self.articlePool= sample(self.AM, self.poolArticleSize)
    # def __init__(
    #     self,
    #     client_index,
    #     train_data_local_dict,
    #     train_data_local_num_dict,
    #     test_data_local_dict,
    #     train_data_num,
    #     device,
    #     args,
    #     model_trainer,
    # ):
    #     self.trainer = model_trainer

    #     self.client_index = client_index

    #     if args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
    #         self.train_data_local_dict = split_data_for_dist_trainers(train_data_local_dict, args.n_proc_in_silo)
    #     else:
    #         self.train_data_local_dict = train_data_local_dict

    #     self.train_data_local_num_dict = train_data_local_num_dict
    #     self.test_data_local_dict = test_data_local_dict
    #     self.all_train_data_num = train_data_num
    #     self.train_local = None
    #     self.local_sample_number = None
    #     self.test_local = None

    #     self.device = device
    #     self.args = args
    #     self.args.device = device

    # def update_model(self, weights):
    #     self.trainer.set_model_params(weights)

    # def update_dataset(self, client_index):
    #     self.client_index = client_index

    #     if self.train_data_local_dict is not None:
    #         if self.args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
    #             self.train_local = self.train_data_local_dict[client_index][self.args.proc_rank_in_silo]
    #         else:
    #             self.train_local = self.train_data_local_dict[client_index]
    #     else:
    #         self.train_local = None

    #     if self.train_data_local_num_dict is not None:
    #         self.local_sample_number = self.train_data_local_num_dict[client_index]
    #     else:
    #         self.local_sample_number = 0

    #     if self.test_data_local_dict is not None:
    #         self.test_local = self.test_data_local_dict[client_index]
    #     else:
    #         self.test_local = None

    #     self.trainer.update_dataset(self.train_local, self.test_local, self.local_sample_number)

    # def train(self, round_idx=None):
    #     self.args.round_idx = round_idx
    #     tick = time.time()

    #     self.trainer.on_before_local_training(self.train_local, self.device, self.args)
    #     self.trainer.train(self.train_local, self.device, self.args)
    #     self.trainer.on_after_local_training(self.train_local, self.device, self.args)

    #     MLOpsProfilerEvent.log_to_wandb({"Train/Time": time.time() - tick, "round": round_idx})
    #     weights = self.trainer.get_model_params()
    #     # transform Tensor to list
    #     return weights, self.local_sample_number

    # def test(self):
    #     # train data
    #     train_metrics = self.trainer.test(self.train_local, self.device, self.args)
    #     train_tot_correct, train_num_sample, train_loss = (
    #         train_metrics["test_correct"],
    #         train_metrics["test_total"],
    #         train_metrics["test_loss"],
    #     )

    #     # test data
    #     test_metrics = self.trainer.test(self.test_local, self.device, self.args)
    #     test_tot_correct, test_num_sample, test_loss = (
    #         test_metrics["test_correct"],
    #         test_metrics["test_total"],
    #         test_metrics["test_loss"],
    #     )

    #     return (
    #         train_tot_correct,
    #         train_loss,
    #         train_num_sample,
    #         test_tot_correct,
    #         test_loss,
    #         test_num_sample,
    #     )

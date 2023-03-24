import copy
import logging
import random
import time
import numpy as np
import torch
import wandb
from .utils import transform_list_to_tensor
from ....core.security.fedml_attacker import FedMLAttacker
from ....core.security.fedml_defender import FedMLDefender
from util_functions import featureUniform, gaussianFeature, dsigmoid, sigmoid

class DisLinUCBAggregator(object):
    def __init__(
        self,
        dimension,
        alpha,
        lambda_,
        delta_,
        threshold,
        worker_num,
        device,
        args,
        server_aggregator,
        AM,

    ):
        self.dimension = dimension
        self.alpha = alpha
        self.lambda_ = lambda_
        self.delta_ = delta_
        self.threshold = threshold
        self.CanEstimateUserPreference = True
        self.clients = {}
        self.A_aggregated = np.zeros((self.dimension, self.dimension))
        self.b_aggregated = np.zeros(self.dimension)
        self.numObs_aggregated = 0
        self.totalCommCost = 0
        self.aggregator = server_aggregator

        self.args = args

        self.worker_num = worker_num
        self.device = device
        self.W_dict = dict()
        self.U_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.aggregator.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.aggregator.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        # self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_one_recieve(self):
        logging.debug("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if self.flag_client_model_uploaded_dict[idx]:
                return True
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return False

    def check_whether_all_receive(self):
        logging.debug("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True


    def aggregate(self):
        start_time = time.time()
        model_W_list = []
        model_U_list = []

        for idx in range(self.worker_num):
            model_W_list.append((self.model_dict[idx][0])) # self.sample_num_dict[idx],
            model_U_list.append((self.model_dict[idx][1]))
            # training_num += self.sample_num_dict[idx]
        logging.info("len of self.model_W_dict[idx] = " + str(len(self.W_dict)))
        logging.info("len of self.model_U_dict[idx] = " + str(len(self.U_dict)))

        Wsyn = self.A_aggregated + sum(model_W_list)
        Usyn = self.b_aggregated + sum(model_U_list)
        
        self.set_global_model_params((Wsyn, Usyn))
        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return (Wsyn, Usyn)


    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [
                client_index for client_index in range(client_num_in_total)
            ]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(
                round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(
                range(client_num_in_total), num_clients, replace=False
            )
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num = len(self.test_global.dataset)
            sample_indices = random.sample(
                range(test_data_num), min(num_samples, test_data_num)
            )
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(
                subset, batch_size=self.args.batch_size
            )
            return sample_testset
        else:
            return self.test_global

    def GetOptimalReward(self, articlePool):		
        maxReward = float('-inf')
        maxx = None
        u = random.choices(population=self.clients, weights=None, k=1)[0]
        for x in articlePool:
            reward = self.getReward(u, x)
        if reward > maxReward:
            maxReward = reward
            maxx = x   
            if self.reward_model == 'linear':
                maxReward = maxReward
        elif self.reward_model == 'sigmoid':
            maxReward = sigmoid(maxReward)
        else:
            raise ValueError
        return maxReward, 

    def getReward(self, user, pickedArticle):
        inner_prod = np.dot(user.theta, pickedArticle.featureVector)
        if self.reward_model == 'linear':
            reward = inner_prod
        elif self.reward_model == 'sigmoid':
            reward = sigmoid(inner_prod)
        else:
            raise ValueError
        return reward
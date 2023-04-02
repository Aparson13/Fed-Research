import time

from fedml.data import split_data_for_dist_trainers
from ...constants import FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL
from ...core.mlops.mlops_profiler_event import MLOpsProfilerEvent
from ..util_functions import featureUniform, gaussianFeature, dsigmoid, sigmoid
import numpy as np

class FedMLTrainer(object):
    def __init__(
        self,
        client_index,
        device,
        args,
        dim,
        lambda_,
        delta_,
        threshold,
        n_articles,
        model_trainer,
    ):
        
        self.d = dim
        self.lambda_ = lambda_
        self.delta_ = delta_
        self.A_local = np.zeros((self.d, self.d))
        self.b_local = np.zeros(self.d)
        self.numObs_local = 0
        self.A_uploadbuffer = np.zeros((self.d, self.d))
        self.b_uploadbuffer = np.zeros(self.d)
        self.numObs_uploadbuffer = 0
        # for computing UCB
        self.AInv = np.linalg.inv(self.A_local+self.lambda_ * np.identity(n=self.d))
        self.UserTheta = np.zeros(self.d)
        self.alpha_t = np.sqrt(self.d * np.log(1 + self.numObs_local)/ (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_) + np.sqrt(self.lambda_)
        self.reward_model = "linear"

    def train(self, A_local, b_local, A_uploadbuffer, b_uploadbuffer, numObs_local, numObs_uploadbuffer, alpha_t, 
              d, articlepool, threshold, device, delta_, lambda_, args):

        maxPTA = float('-inf')
        articlePicked = None
        numerator = np.linalg.det(A_local+lambda_ * np.identity(n=d))
        denominator = np.linalg.det(A_local-A_uploadbuffer+self.lambda_ * np.identity(n=d))

        while(np.log(numerator/denominator)*(self.numObs_uploadbuffer) < threshold and self.sync == False):
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
            numObs_local += 1

            A_uploadbuffer += np.outer(articlePicked.featureVector, articlePicked.featureVector)
            b_uploadbuffer += articlePicked.featureVector * reward
            numObs_uploadbuffer += 1

            AInv = np.linalg.inv(A_local+lambda_ * np.identity(n=d))
            UserTheta = np.dot(AInv, b_local)

            self.alpha_t = self.NoiseScale * np.sqrt(
                d * np.log(1 + (numObs_local) / (d * lambda_)) + 2 * np.log(1 / delta_)) + np.sqrt(
                lambda_)
            
            numerator = np.linalg.det(self.A_local+self.lambda_ * np.identity(n=d))
            denominator = np.linalg.det(A_local-A_uploadbuffer+lambda_ * np.identity(n=self.d))

        return A_uploadbuffer, b_uploadbuffer


    def getUCB(self, alpha, article_FeatureVector):
        if alpha == -1:
            alpha = self.alpha_t

        mean = np.dot(self.UserTheta, article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv), article_FeatureVector))
        pta = mean + alpha * var
        return pta
    

    def update_dataset(self, global_model_params, client_index):
        articlePicked_FeatureVector = global_model_params[0]
        click = global_model_params[1]
        self.A_local += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b_local += articlePicked_FeatureVector * click
        self.numObs_local += 1

        self.A_uploadbuffer = np.zeros((self.dimension, self.dimension))
        self.b_uploadbuffer = np.zeros(self.dimension)
        self.numObs_uploadbuffer = 0

        self.AInv = np.linalg.inv(self.A_local+self.lambda_ * np.identity(n=self.d))
        self.UserTheta = np.dot(self.AInv, self.b_local)

        self.alpha_t = self.NoiseScale * np.sqrt(
        self.d * np.log(1 + (self.numObs_local) / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_)) + np.sqrt(
        self.lambda_)
        self.trainer.sync = False

    def getTheta(self):
        return self.UserTheta
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

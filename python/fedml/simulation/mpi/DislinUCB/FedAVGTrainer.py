from .utils import transform_tensor_to_list
import numpy as np
import copy

class DisLinUCBTrainer(object):
    def __init__(
        self,
        client_index,
        device,
        args,
        ################################# Start
        d,
        lambda_,
        delta_,
        A_local,
        B_local,
        numObs_local,
        A_uploadbuffer,
        B_uploadbuffer,
        numObs_uploadbuffer,
        AInv, 
        UserTheta,
        alpha_t,
        ################################# End
        model_trainer,
    ):
        self.trainer = model_trainer
        self.client_index = client_index
        self.train_local = None
        self.local_sample_number = None
        self.test_local = None

        self.device = device
        self.args = args

        ################################### Start

        self.d = args.featureDimension
        self.lambda_ = args.lambda_
        self.delta_ = args.delta_
        self.A_local = np.zeros((self.d, self.d))
        self.B_local = np.zeros(self.d)
        self.numObs_local = 0
        self.A_uploadbuffer = np.zeros((self.d, self.d))
        self.b_uploadbuffer = np.zeros(self.d)
        self.numObs_uploadbuffer = 0
        # for computing UCB
        self.AInv = np.linalg.inv(self.A_local+self.lambda_ * np.identity(n=self.d))
        self.UserTheta = np.zeros(self.d)
        self.alpha_t = np.sqrt(self.d * np.log(1 + self.numObs_local)/ (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_) + np.sqrt(self.lambda_)

        ##################################### End
    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    

    # def update_dataset(self, client_index):
    #     self.client_index = client_index

    #     if self.train_data_local_dict is not None:
    #         self.train_local = self.train_data_local_dict[client_index]
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

    def train(self, round_idx=None):
        self.args.round_idx = round_idx
        self.trainer.train(self.train_local, self.device, self.args)

        weights = self.trainer.get_model_params()

        return weights, self.local_sample_number

    def test(self):
        # train data
        train_metrics = self.trainer.test(self.train_local, self.device, self.args)
        train_tot_correct, train_num_sample, train_loss = (
            train_metrics["test_correct"],
            train_metrics["test_total"],
            train_metrics["test_loss"],
        )

        # test data
        test_metrics = self.trainer.test(self.test_local, self.device, self.args)
        test_tot_correct, test_num_sample, test_loss = (
            test_metrics["test_correct"],
            test_metrics["test_total"],
            test_metrics["test_loss"],
        )

        return (
            train_tot_correct,
            train_loss,
            train_num_sample,
            test_tot_correct,
            test_loss,
            test_num_sample,
        )
    
    ##################################################### Start
    def localUpdate(self, articlePicked_FeatureVector, click):
        self.A_local += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b_local += articlePicked_FeatureVector * click
        self.numObs_local += 1

        self.A_uploadbuffer += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b_uploadbuffer += articlePicked_FeatureVector * click
        self.numObs_uploadbuffer += 1

        self.AInv = np.linalg.inv(self.A_local+self.lambda_ * np.identity(n=self.d))
        self.UserTheta = np.dot(self.AInv, self.b_local)

        self.alpha_t = self.NoiseScale * np.sqrt(
            self.d * np.log(1 + (self.numObs_local) / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_)) + np.sqrt(
            self.lambda_)
        
    def getTheta(self):
        return self.UserTheta
    
    def syncRoundTriggered(self, threshold):
        numerator = np.linalg.det(self.A_local+self.lambda_ * np.identity(n=self.d))
        denominator = np.linalg.det(self.A_local-self.A_uploadbuffer+self.lambda_ * np.identity(n=self.d))
        return np.log(numerator/denominator)*(self.numObs_uploadbuffer) >= threshold
    
    ####################################################### End
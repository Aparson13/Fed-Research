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
        dim,
        lambda_,
        delta_,
        threshold,
        ################################# End
        model_trainer,
    ):
        self.trainer = model_trainer
        self.client_index = client_index
        self.device = device
        self.args = args

        ################################### Start

        self.d = dim
        self.lambda_ = lambda_
        self.delta_ = delta_
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
        self.reward_model = "linear"

        ##################################### End
    # def update_model(self, weights):
    #     self.trainer.set_model_params(weights)

    

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
       

    def train(self, round_idx=None):
        self.args.round_idx = round_idx
        A_uploadbuffer, b_uploadbuffer = self.trainer.train(self.A_local, self.B_local, self.A_upload, self.b_upload, self.numObs_local, self.AInv, self.UserTheta,
                           self.alpha_t, self.d, self.AM, self.device, self.args)


        return A_uploadbuffer, b_uploadbuffer
        # Need to change the prior function and I guess make it more similar to how they do it in simulation distributed
        # Since this is the trainer method, just focus on training one user and the rest should take care of the rest
        # Try making it so that when getting model params, we get W and U back instead of the weights
        # Also try making the train a loop until det(V) various and then the train method ends and changes the model parameters

        
    def getTheta(self):
        return self.UserTheta
    
    
    
   
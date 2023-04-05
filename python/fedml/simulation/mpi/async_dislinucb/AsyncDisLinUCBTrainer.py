import numpy as np
from random import sample, shuffle
from util_function import sigmoid

class AsyncDisLinUCBTrainer(object):
    def __init__(
        self,
        client_index,
        dimension,
        lamda,
        delta,
        noise,
        local_articles_num_dict,
        local_articles_dict,
        articles_num,
        device,
        args,
        model_trainer,
    ):
        self.trainer = model_trainer

        self.client_index = client_index
        # self.train_data_local_dict = train_data_local_dict
        # self.train_data_local_num_dict = train_data_local_num_dict
        # self.test_data_local_dict = test_data_local_dict
        # self.all_train_data_num = train_data_num
        self.articlePool = sample(self.local_articles, self.poolArticleSize)
        self.lamda = lamda
        self.delta = delta
        self.noise = noise
        self.local_articles_num_dict = local_articles_num_dict
        self.local_articles_dict = local_articles_dict
        self.articles_num = articles_num
        self.d = dimension
        # self.train_local = None
        self.local_sample_number = None
        # self.test_local = None4
        self.local_article = None
        self.A_local = np.zeros((self.d, self.d))  #lambda_ * np.identity(n=self.d)
        self.b_local = np.zeros(self.d)

        self.A_uploadbuffer = np.zeros((self.d, self.d))
        self.b_uploadbuffer = np.zeros(self.d)

        self.AInv = np.linalg.inv(self.A_local+self.lambda_ * np.identity(n=self.d))
        self.UserTheta = np.zeros(self.d)

        self.device = device
        self.args = args

    def update_model(self, articles):
        self.trainer.set_model_params(articles)

    # def update_dataset(self, client_index):
    #     self.client_index = client_index
    #     self.train_local = self.train_data_local_dict[client_index]
    #     self.local_sample_number = self.train_data_local_num_dict[client_index]
    #     self.test_local = self.test_data_local_dict[client_index]

    # def train(self, round_idx=None):
    #     self.args.round_idx = round_idx
    #     self.trainer.train(self.train_local, self.device, self.args)

    #     weights = self.trainer.get_model_params()

    #     return weights, self.local_sample_number

    def getReward(self, pickedArticle):
        inner_prod = np.dot(self.UserTheta, pickedArticle.featureVector)
        if self.reward_model == 'linear':
            reward = inner_prod
        elif self.reward_model == 'sigmoid':
            reward = sigmoid(inner_prod)
        else:
            raise ValueError
        return reward
    
    def uploadTrigger(self):
        return self.trainer.uploadTrigger()
    
    def train(self, round_idx = None):
        self.args.round_inx = round_idx
        pickedArticle = self.trainer.decide(self.articlePool)
        reward = self.getReward(pickedArticle)
        self.trainer.train(pickedArticle, pickedArticle.featureVector, self.device, self.args, reward)

        W, U = self.trainer.get_model_params()


        return W, U, self.local_sample_number

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

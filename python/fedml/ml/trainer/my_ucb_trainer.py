import numpy as np
from random import sample, shuffle

class UCBTrainer(object):


    def __init__(self, dimension, lamda, delta, alpha, noise,local_articles, gammaU, args):
        #self.model = model
        self.local_articles = local_articles
        self.id = 0
        self.args = args
        self.articlePool = sample(self.local_articles, self.poolArticleSize)
        self.lambda_ = lamda
        self.delta_ = delta
        self.noise = noise
        self.d = dimension
        self.alpha_ = alpha
        self.gammaU = gammaU
        # self.local_train_dataset = None
        # self.local_test_dataset = None
        # self.local_sample_number = 0
        self.A_local = np.zeros((self.d, self.d))  #lambda_ * np.identity(n=self.d)
        self.b_local = np.zeros(self.d)
        self.numObs_local = 0

        # aggregated sufficient statistics recently downloaded
        self.A_uploadbuffer = np.zeros((self.d, self.d))
        self.b_uploadbuffer = np.zeros(self.d)

        self.AInv = np.linalg.inv(self.A_local+self.lambda_ * np.identity(n=self.d))
        self.UserTheta = np.zeros(self.d)

        self.alpha_t = self.NoiseScale * np.sqrt(
            self.d * np.log(1 + (self.numObs_local) / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_)) + np.sqrt(
            self.lambda_)

        
        
    def set_id(self, trainer_id):
        self.id = trainer_id

    # def update_dataset(self, local_train_dataset, local_test_dataset, local_sample_number):
    #     self.local_train_dataset = local_train_dataset
    #     self.local_test_dataset = local_test_dataset
    #     self.local_sample_number = local_sample_number

    def update_dataset(self, A_local, B_local):
        self.A_local = A_local
        self.B_local = B_local



    def get_model_params(self):
        #return self.model.cpu().state_dict()
        return self.A_uploadbuffer, self.b_uploadbuffer

    def set_model_params(self, articles):
        #self.model.load_state_dict(model_parameters)
        self.local_articles = articles

    def decide(self, pool_articles): #ClientID
        # if clientID not in self.clients:
        #     self.clients[clientID] = LocalClient(self.dimension, self.lambda_, self.delta_, self.NoiseScale)
        #     self.A_downloadbuffer[clientID] = copy.deepcopy(self.A_aggregated)
        #     self.b_downloadbuffer[clientID] = copy.deepcopy(self.b_aggregated)
        #     self.numObs_downloadbuffer[clientID] = copy.deepcopy(self.numObs_aggregated)

        maxPTA = float('-inf')
        articlePicked = None

        for x in pool_articles:
            x_pta = self.getUCB(self.alpha, x.featureVector)
            # pick article with highest UCB score
            if maxPTA < x_pta:
                articlePicked = x
                maxPTA = x_pta

        return articlePicked
    
    def getUCB(self, alpha, article_FeatureVector):
        if alpha == -1:
            alpha = self.alpha_t

        mean = np.dot(self.UserTheta, article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv), article_FeatureVector))
        pta = mean + alpha * var
        return pta
    
    def uploadCommTriggered(self):
        numerator = np.linalg.det(self.A_local+self.lambda_ * np.identity(n=self.d))
        denominator = np.linalg.det(self.A_local-self.A_uploadbuffer+self.lambda_ * np.identity(n=self.d))
        return numerator/denominator > self.gammaU

    def train(self, articlePicked_FeatureVector, device, args, click): #Click stand for reward
        # model = self.model

        # model.to(device)
        # model.train()

        # train and update
        #criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
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
        
    
        
        

   
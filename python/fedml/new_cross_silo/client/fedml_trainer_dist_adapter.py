import logging

from fedml.constants import FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL
from .fedml_trainer import FedMLTrainer
from ...ml.trainer.trainer_creator import create_model_trainer
from ...ml.engine import ml_engine_adapter
from .ArticleManager import ArticleManager
from util_functions import featureUniform, gaussianFeature, dsigmoid, sigmoid

class TrainerDistAdapter:
    def __init__(
        self,
        client_rank,
        device,
        args,
        dim,
        lambda_,
        delta_,
        threshold,
        n_articles,
        model_trainer,
    ):
        
        AM = ArticleManager(dim, n_articles, gaussianFeature, argv={'l2_limit': 1}, ArticleGroups=0)
        self.AM = AM.simulateArticlePool()
        self.trainer = model_trainer
        self.client_rank = client_rank
        self.device = device
        self.args = args

        # ml_engine_adapter.model_to_device(args, model, device)

        # if args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
        #     self.process_group_manager, model = ml_engine_adapter.model_ddp(args, model, device)

        # if model_trainer is None:
        #     model_trainer = create_model_trainer(model, args)
        # else:
        #     model_trainer.model = model

        client_index = client_rank - 1

        model_trainer.set_id(client_index)

        logging.info("Initiating Trainer")
        trainer = self.get_trainer(
            client_index,
            device,
            args,
            dim,
            lambda_,
            delta_,
            threshold,
            n_articles,
            model_trainer,
        )
        self.client_index = client_index
        self.client_rank = client_rank
        self.device = device
        self.trainer = trainer
        self.args = args

    def get_trainer(
        self,
        client_index,
        dim,
        device,
        args,
        lambda_,
        delta_,
        threshold,
        n_articles,
        ):

        return FedMLTrainer(
        client_index,
        dim,
        device,
        args,
        dim,
        lambda_,
        delta_,
        threshold,
        n_articles,
    
        )

    def train(self, round_idx):
        weights, local_sample_num = self.trainer.train(round_idx)
        return weights, local_sample_num

    # def update_model(self, model_params):
    #     self.trainer.update_model(model_params)

    def update_dataset(self, client_index=None):
        _client_index = client_index or self.client_index
        self.trainer.update_dataset(int(_client_index))

    def cleanup_pg(self):
        if self.args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
            logging.info(
                "Cleaningup process group for client %s in silo %s"
                % (self.args.proc_rank_in_silo, self.args.rank_in_node)
            )
            self.process_group_manager.cleanup()

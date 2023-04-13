import logging

from fedml.constants import FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL
from .fedml_trainer import FedMLTrainer
from ...ml.trainer.trainer_creator import create_model_trainer
from ...ml.engine import ml_engine_adapter
from .ArticleManager import ArticleManager
from ..util_functions import featureUniform, gaussianFeature, dsigmoid, sigmoid

class TrainerDistAdapter:
    def __init__(
        self,
        args,
        client_rank,
        device,
        dim,
        lambda_,
        alpha,
        delta_,
        threshold,
        n_articles,
        model_trainer,
    ):
        self.n_articles = n_articles
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

        # model_trainer.set_id(client_index)

        logging.info("Initiating Trainer")
        self.trainer = self.get_trainer(
            args,
            client_index,
            device,
            dim,
            lambda_,
            alpha,
            delta_,
            threshold,
            n_articles,
        )
        # self.client_index = client_index
        # self.client_rank = client_rank
        # self.device = device
        # self.trainer = trainer
        # self.args = args

    def get_trainer(
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

        return FedMLTrainer(
        args,
        client_index,
        device,
        dim,
        lambda_,
        alpha,
        delta_,
        threshold,
        n_articles,
        )

    def train(self, round_idx):
        weights, local_sample_num = self.trainer.train(round_idx)
        return weights, local_sample_num

    # def update_model(self, model_params):
    #     self.trainer.update_model(model_params)

    def update_dataset(self, model_params, client_index=None):
        _client_index = client_index or self.client_index
        self.trainer.update_dataset(model_params, int(_client_index))

    def cleanup_pg(self):
        if self.args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
            logging.info(
                "Cleaningup process group for client %s in silo %s"
                % (self.args.proc_rank_in_silo, self.args.rank_in_node)
            )
            self.process_group_manager.cleanup()

from .client import client_initializer
from ..core import ClientTrainer


class FedMLCrossSiloClient:
    def __init__(self, args, device, dataset, model, model_trainer: ClientTrainer = None):

        if args.federated_optimizer == "DisLinUCB":
            dimension = args.dimension
            n_articles = args.n_articles
            alpha = args.alpha
            lambda_ = args.lambda_
            delta_ = args.delta_
            threshold = args.threshold
            client_initializer.init_client(
                args,
                device,
                args.comm,
                args.rank,
                args.worker_num,
                model,
                dimension,
                alpha,
                lambda_,
                delta_,
                threshold,
                n_articles,
                model_trainer,
                )

        elif args.federated_optimizer == "LSA":
            from .lightsecagg.lsa_fedml_api import FedML_LSA_Horizontal

            FedML_LSA_Horizontal(
                args,
                args.rank,
                args.worker_num,
                args.comm,
                device,
                dataset,
                model,
                model_trainer=model_trainer,
                preprocessed_sampling_lists=None,
            )
        elif args.federated_optimizer == "SA":
            from .secagg.sa_fedml_api import FedML_SA_Horizontal

            FedML_SA_Horizontal(
                args,
                args.rank,
                args.worker_num,
                args.comm,
                device,
                dataset,
                model,
                model_trainer=None,
                preprocessed_sampling_lists=None,
            )
        else:
            raise Exception("Exception")

    def run(self):
        pass

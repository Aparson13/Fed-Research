from fedml.core import ServerAggregator


class FedMLCrossSiloServer:
    def __init__(self, args, device, dataset, model, server_aggregator: ServerAggregator = None):
        print(args.federated_optimizer)
        if args.federated_optimizer == "DisLinUCB":
            from fedml.cross_silo.server import server_initializer

            [
                dimension,
                alpha,
                lambda_,
                delta_,
                threshold,
            ] = dataset
            server_initializer.init_server(
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
                server_aggregator,
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
                model_trainer=None,
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

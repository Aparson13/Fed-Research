from .DisLinUCBAggregator import DisLinUCBAggregator
from .DisLinUCBTrainer import DisLinUCBTrainer
from .DinLinUCBClientManager import DisLinUCBClientManager
from .DisLinUCBServerManager import DisLinUCBServerManager
from .ArticleManager import ArticleManager
from ....core import ClientTrainer, ServerAggregator
from ....core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
from ....core.security.fedml_attacker import FedMLAttacker
from ....core.security.fedml_defender import FedMLDefender
from ....ml.aggregator.aggregator_creator import create_server_aggregator
from ....ml.trainer.trainer_creator import create_model_trainer
from util_functions import featureUniform, gaussianFeature, dsigmoid, sigmoid

# Need to make it available that the clients contact the server when they are ready but I think after train method over, it just goes aheads and contacts the server
def FedML_DisLinUCB_distributed(
    args,
    process_id,
    worker_number,
    comm,
    device,
    dataset,
    model,
    client_trainer: ClientTrainer = None,
    server_aggregator: ServerAggregator = None,
):
    [   
        dimension,
        n_articles,
        alpha,
        lambda_,
        delta_,
        threshold,
    ] = dataset

    
    FedMLAttacker.get_instance().init(args)
    FedMLDefender.get_instance().init(args)
    FedMLDifferentialPrivacy.get_instance().init(args)

    if process_id == 0:
        init_server(
            args,
            device,
            comm,
            process_id,
            worker_number,
            model,
            dimension,
            alpha,
            lambda_,
            delta_,
            threshold,
            server_aggregator,
        )
    else:
        init_client(
            args,
            device,
            comm,
            process_id,
            worker_number,
            model,
            dimension,
            alpha,
            lambda_,
            delta_,
            threshold,
            n_articles,
            client_trainer,
        )


def init_server(
    args,
    device,
    comm,
    rank,
    size,
    model,
    dimension,
    alpha,
    lambda_,
    delta_,
    threshold,
    server_aggregator,
):
    if server_aggregator is None:
        server_aggregator = create_server_aggregator(model, args)
    server_aggregator.set_id(-1)

    # aggregator
    worker_num = size - 1
    aggregator = DisLinUCBAggregator(
        dimension,
        alpha,
        lambda_,
        delta_,
        threshold,
        worker_num,
        device,
        args,
        server_aggregator,
    )

    # start the distributed training
    backend = args.backend
    server_manager = DisLinUCBServerManager(args, aggregator, comm, rank, size, backend)
    server_manager.send_init_msg()
    server_manager.run()


def init_client(
    args,
    device,
    comm,
    process_id,
    size,
    model,
    dim,
    alpha,
    lambda_,
    delta_,
    threshold,
    n_articles,
    model_trainer=None,
    
):
    client_index = process_id - 1
    if model_trainer is None:
        model_trainer = create_model_trainer(model, args)
    model_trainer.set_id(client_index)
    backend = args.backend
    trainer = DisLinUCBTrainer(
        client_index,
        dim,
        device,
        args,
        dim,
        lambda_,
        delta_,
        threshold,
        n_articles,
        model_trainer,
    )
    client_manager = DisLinUCBClientManager(args, trainer, comm, process_id, size, backend)
    client_manager.run()

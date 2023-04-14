from .fedml_aggregator import FedMLAggregator
from .fedml_server_manager import FedMLServerManager
from ...ml.aggregator.aggregator_creator import create_server_aggregator


def init_server(
    args,
    device,
    comm,
    rank,
    worker_num,
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
    server_aggregator.set_id(0)

    # aggregator
    aggregator = FedMLAggregator(
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
    server_manager = FedMLServerManager(args, aggregator, comm, rank, worker_num, backend)
    server_manager.run()

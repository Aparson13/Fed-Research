from ....core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
from ....core.security.fedml_attacker import FedMLAttacker
from ....core.security.fedml_defender import FedMLDefender
#from ....ml.trainer.trainer_creator import create_model_trainer
from ....ml.trainer.ucb_trainer_creator import create_model_trainer
from .AsyncDisLinUCBTrainer import AsyncDisLinUCBTrainer
from .AsyncDisLinUCBClientManager import AsyncDisLinUCBClientManager
from .AsyncDisLinUCBAggregator import AsyncDisLinUCBAggregator
from .AsyncDisLinUCBServerManager import AsyncDisLinUCBServerManager

def FedML_AsyncDisLinUCB_distributed(
    args, process_id, worker_number, comm, device, dataset, model, model_trainer=None, preprocessed_sampling_lists=None,
):
    # [
    #     train_data_num,
    #     test_data_num,
    #     train_data_global,
    #     test_data_global,
    #     train_data_local_num_dict,
    #     train_data_local_dict,
    #     test_data_local_dict,
    #     class_num,
    # ] = dataset
    
    #This is not a supervised learning, so I guess we don't have to use a training and testing set?
    #Instead, we use articles(bandit) here to replace the training data
    [
        dimension,
        lamda,
        delta,
        alpha,
        noise,
        articles_num,
        local_articles,
        global_articles,
        local_articles_num_dict,
        local_articles_dict
    ] = dataset

    FedMLAttacker.get_instance().init(args)
    FedMLDefender.get_instance().init(args)
    FedMLDifferentialPrivacy.get_instance().init(args)

    if process_id == 0:
        # init_server(
        #     args,
        #     device,
        #     comm,
        #     process_id,
        #     worker_number,
        #     model,
        #     train_data_num,
        #     train_data_global,
        #     test_data_global,
        #     train_data_local_dict,
        #     test_data_local_dict,
        #     train_data_local_num_dict,
        #     model_trainer,
        #     preprocessed_sampling_lists,
        # )
        init_server(
            args,
            device,
            comm,
            process_id,
            worker_number,
            model,
            articles_num,
            global_articles,
            model_trainer,
            preprocessed_sampling_lists
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
            lamda,
            delta,
            alpha,
            noise,
            articles_num,
            local_articles,
            global_articles,
            local_articles_num_dict,
            local_articles_dict,
            model_trainer,
        )


def init_server(
    args,
    device,
    comm,
    rank,
    size,
    model,
    articles_num,
    global_articles,
    model_trainer,
    preprocessed_sampling_lists=None,
):
    if model_trainer is None:
        model_trainer = create_model_trainer(model, args)
    model_trainer.set_id(-1)

    # aggregator
    worker_num = size - 1
    # aggregator = AsyncFedAVGAggregator(
    #     train_data_global,
    #     test_data_global,
    #     train_data_num,
    #     train_data_local_dict,
    #     test_data_local_dict,
    #     train_data_local_num_dict,
    #     worker_num,
    #     device,
    #     args,
    #     model_trainer,
    # )

    aggregator = AsyncDisLinUCBAggregator(
        articles_num,
        global_articles,
        worker_num,
        device,
        args,
        model_trainer,
    )

    # start the distributed training
    backend = args.backend
    if preprocessed_sampling_lists is None:
        server_manager = AsyncDisLinUCBServerManager(args, aggregator, comm, rank, size, backend)
    else:
        server_manager = AsyncDisLinUCBServerManager(
            args,
            aggregator,
            comm,
            rank,
            size,
            backend,
            is_preprocessed=True,
            preprocessed_client_lists=preprocessed_sampling_lists,
        )
    server_manager.send_init_msg()
    server_manager.run()


def init_client(
    args,
    device,
    comm,
    process_id,
    size,
    model,
    dimension,
    lamda,
    delta,
    alpha,
    noise,
    articles_num,
    local_articles,
    global_articles,
    local_articles_num_dict,
    local_articles_dict,
    model_trainer=None,
):
    client_index = process_id - 1
    # if model_trainer is None:
    #     model_trainer = create_model_trainer(model, args)
    if model_trainer is None:
        model_trainer = create_model_trainer(dimension,lamda, delta, alpha, noise, local_articles, articles_num, args)
    model_trainer.set_id(client_index)
    backend = args.backend
    trainer = AsyncDisLinUCBTrainer(
        client_index,
        dimension,
        lamda,
        delta,
        alpha,
        noise,
        local_articles_num_dict,
        local_articles_dict,
        articles_num,
        device,
        args,
        model_trainer,
    )
    client_manager = AsyncDisLinUCBClientManager(args, trainer, comm, process_id, size, backend)
    client_manager.run()

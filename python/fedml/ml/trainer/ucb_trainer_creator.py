from .my_ucb_trainer import UCBTrainer

def create_model_trainer(dimension,lamda, delta, alpha, noise, local_articles, args):
    # if args.dataset == "stackoverflow_lr":
    #     model_trainer = ModelTrainerTAGPred(model, args)
    # elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
    #     model_trainer = ModelTrainerNWP(model, args)
    # else:  # default model trainer is for classification problem
    #     model_trainer = ModelTrainerCLS(model, args)
    # return model_trainer
    create_model_trainer = UCBTrainer(dimension,lamda, alpha, delta, noise, local_articles,  args)

    return create_model_trainer

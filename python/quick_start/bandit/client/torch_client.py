import fedml
from fedml import FedMLRunner

if __name__ == "__main__":
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset = 1
    output_dim = 1

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()

import numpy as np
import torch

from experiments import Experiment, PretrainExperiment
from settings import Settings, parse_arguments


def main():
    # get the settings from the command line
    ss = Settings(parse_arguments())
    ss.make_dirs()

    torch.manual_seed(ss.args.seed)
    torch.cuda.manual_seed(ss.args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    np.random.seed(ss.args.seed)
    exp_name = 'general'


    print(f"Running {exp_name} experiment...")
    exp = PretrainExperiment(ss)


    # Run experiment
    exp.run_experiment()


if __name__ == '__main__':
    main()
    pass

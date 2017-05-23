import argparse
import os
from os.path import join as pjoin

from distributions.MixSkewStudentT import *

def build_argparser():
    DESCRIPTION = ("Train a Mixture of Skew Multivariate Student-t Distributions.")
    p = argparse.ArgumentParser(description=DESCRIPTION)

    dataset = p.add_argument_group("Experiment options")
    dataset.add_argument('--dataset', default="test1", choices=["test1"],
                         help="Input dataset. Default:%(default)s")

    model = p.add_argument_group("Model options")
    model.add_argument('--nb-components', type=int, default=2,
                         help='Number of components in mixture model. Default:%(default)s')

    training = p.add_argument_group("Training options")
    training.add_argument('--batch-size', type=int, default=100,
                          help='size of the batch to use when training the model. Default: %(default)s.')
    training.add_argument('--max-epoch', type=int, metavar='N', default=2000,
                          help='train for a maximum of N epochs. Default: %(default)s')

    general = p.add_argument_group("General arguments")
    general.add_argument('--experiment-dir', default="./experiments/",
                         help='name of the folder where to save the experiment. Default: %(default)s.')

    return p


if __name__ == '__main__':
    random_seed = 1234

    # get command-line args
    parser = build_argparser()
    args = parser.parse_args()
    args_dict = vars(args)
    args_string = ''.join('{}_{}_'.format(key, val) for key, val in sorted(args_dict.items()) if key not in ['experiment_dir','lookahead','max_epoch'])[:-1]

    # Check the parameters are correct.
    if args.nb_components < 1:
        raise ValueError("Number of components must be a positive integer.")


    # DATA PARAMS
    # Create datasets and experiments folders is needed.
    dataset_dir = mkdirs("./datasets")
    mkdirs(args.experiment_dir)
    dataset = pjoin(dataset_dir, args.dataset + ".npz")
    print "Datasets dir: {}".format(os.path.abspath(dataset_dir))
    print "Experiment dir: {}".format(os.path.abspath(args.experiment_dir))

    model = MixSkewStudentT( nb_components=args.nb_components )


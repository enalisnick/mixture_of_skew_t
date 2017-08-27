import argparse
import random
import os
from os.path import join as pjoin

from distributions.MixSkewStudentT import *

def build_argparser():
    DESCRIPTION = ("Train a Mixture of Skew Multivariate Student-t Distributions.")
    p = argparse.ArgumentParser(description=DESCRIPTION)

    model = p.add_argument_group("Model options")
    model.add_argument('--dimensionality', type=int, default=2,
                         help='Dimensionality of mixture. Default:%(default)s')
    model.add_argument('--nb-components', type=int, default=2,
                         help='Number of components in mixture model. Default:%(default)s')

    training = p.add_argument_group("Training options")
    training.add_argument('--batch-size', type=int, default=100,
                          help='size of the batch to use when training the model. Default: %(default)s.')
    training.add_argument('--max-epoch', type=int, metavar='N', default=2000,
                          help='train for a maximum of N epochs. Default: %(default)s')

    general = p.add_argument_group("General arguments")
    general.add_argument('--experiment-dir', default="./experiments/simulation",
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
    if args_dict['nb_components'] < 1:
        raise ValueError("Number of components must be a positive integer.")


    # DATA PARAMS
    ### true model
    means = [np.array([3.,1.]), np.array([0.,0.])]
    covs = [np.array([[2., 0.],[0., 3.]]), np.eye(args_dict['dimensionality'])]
    skews = [np.array([1.2, 4.5]), np.array([2., 6.])]
    dfs = [2,3]
    n_data = 1000

    ### simulate data from true model
    true_model = MixSkewStudentT(mus=[np.array([3.,1.]), np.array([0.,0.])], Sigmas=[np.array([[2., 0.],[0., 3.]]), np.eye(args_dict['dimensionality'])], \
                                     nb_components=args_dict['nb_components'], deltas=skews, dfs=dfs)
    data = true_model.draw_sample(n_data)

    print data

    # INIT MODEL FOR LEARNING
    #model = MixSkewStudentT( nb_components=args.nb_components )

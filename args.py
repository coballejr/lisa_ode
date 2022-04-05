from argparse import ArgumentParser
from pathlib import Path
import random
import torch
import numpy as np
from typing import Tuple

class Parser(ArgumentParser):

    def __init__(self):
        super(Parser,self).__init__(description = 'Read')
        self.add_argument('--model', type=str, default="fc", choices=['fc', 'fourier', 'siren'], help='neural network model to use')
        self.add_argument('--loss', type=str, default="standard",choices=['standard', 'lie_scale', 'lie_trans', 'lie_all'], help='loss function to use')
        self.add_argument('--symm_method', type=str, default='full',
                          choices=['approx', 'full'], help='apply symms with first-order approx or use full diffeomorphism.')
        self.add_argument('--experiment', type=str, default='seperable', choices=['seperable'], help='ode to learn')

        # data
        self.add_argument('--nfield', type=int, default = 5000, help="number of training data")
        self.add_argument('--ninit', type=int, default = 500, help="number of initial training data")
        self.add_argument('--ntest', type=int, default= 500, help="number of training data")
        self.add_argument('--train-field-batch', type=int, default= 500, help='field batch size for training')
        self.add_argument('--train-init-batch', type=int, default= 50, help=' batch size for training')
        self.add_argument('--test-batch', type=int, default = 32, help='batch size for testing')

        # training
        self.add_argument('--epoch-start', type=int, default=0, help='epoch to start at, will load pre-trained network')
        self.add_argument('--epochs', type=int, default=2000, help='number of epochs to train')
        self.add_argument('--lr', type=float, default= 1e-3, help='ADAM learning rate')
        self.add_argument('--seed', type=int, default=12345, help='manual seed used in PyTorch and Numpy')
        self.add_argument('--lambda_pde', type=float, default=1.0, help='lambda pde param in lie loss')
        self.add_argument('--lambda_init', type=float, default=1.0, help='lambda bndry param in lie loss')
        self.add_argument('--eps_range', type=float, nargs=2, default=(-2.5, -0.5), help= 'range of epsilon values for lie symms')
        self.add_argument('--neps', type=int, default=1, help= 'number of lie symmetries to train with')
        self.add_argument('--u0', type=float, default= 1, help="initial value in \
                          seperable ode.")
        self.add_argument('--test_symm_choice', type = str, default ='translation', choices = ['translation', 'scaling'],
                          help = 'which equivariance loss for training with both symms.')

        # logging
        self.add_argument('--plot-freq', type=int, default= 100, help='how many epochs to wait before plotting test output')
        self.add_argument('--test-freq', type=int, default= 100, help='how many epochs to test the model')
        self.add_argument('--ckpt-freq', type=int, default=100, help='how many epochs to wait before saving the model')
        self.add_argument('--notes', type=str, default="", help='notes')

    def parse(self):
            args = self.parse_args()

            args.run_dir = Path('./outputs') / \
            f'{args. model}_ntrain{args.nfield}_batch{args.train_field_batch}_{args.loss}'

            if len(args.notes) > 0:
                folder_name = args.run_dir.name + f'{args.notes}'
                args.run_dir = args.run_dir.parent / folder_name

            args.ckpt_dir = args.run_dir / "checkpoints"
            args.pred_dir = args.run_dir / "predictions"
            for path in (args.run_dir, args.ckpt_dir, args.pred_dir):
                Path(path).mkdir(parents=True, exist_ok=True)

            # Set random seed
            if args.seed is None:
                args.seed = random.randint(1, 10000)
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            np.random.seed(seed=args.seed)

            return args



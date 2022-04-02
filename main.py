import torch
from args import Parser
from data.loaders import create_training_loader, create_eval_loader
from models import FullyConnected
from loss import PINNTrainingLoss, EvalLoss
from lisa.lie_field_torch import Scaling, Identity
from viz import plot_prediction

if __name__ == '__main__':

    # parse args
    args = Parser().parse()

    # select model
    if args.model == 'fc':
        mod = FullyConnected()

    # define symmetries
    if args.loss.lower() == 'standard':
        symms = (Identity(),)
    elif args.loss.lower() == 'lie':
        if args.experiment.lower() == 'seperable':
            symms = (Identity(), Scaling())

        else:
            raise NotImplementedError('Invalid experiment string.')

    else:
        raise NotImplementedError('Invalid loss string.')

    # define losses
    training_loss_step = PINNTrainingLoss(mod,
                                          args.lambda_pde,
                                          args.lambda_init)
    eval_loss_step = EvalLoss(mod)

    # create dataloaders
    training_loader = create_training_loader(args.nfield,
                                             args.ninit,
                                             args.u0,
                                             args.train_field_batch,
                                             args.train_init_batch)
    testing_loader = create_eval_loader(args.ntest,
                                        args.ntest//10,
                                        args.u0,
                                        args.test_batch)

    # set optimizers
    optim = torch.optim.Adam(mod.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma = 0.999999)

    # training loop
    iteration = 0
    train_losses = []
    eval_losses = []
    for epoch in range(1, args.epochs+1):
        training_loss = 0
        for mbi, (field_data, init_data) in enumerate(training_loader):
           optim.zero_grad()
           # forward
           for symm in symms:
                loss = training_loss_step(field_data, init_data, symm = symm,
                                          symm_method = args.symm_method, eps = args.eps)
                training_loss += loss.detach() / (len(training_loader)*len(symms))

                # backward
                loss.backward()
                optim.step()

        scheduler.step()
        train_losses.append([epoch*len(training_loader), training_loss.cpu()])

        # validation
        if epoch % args.test_freq == 0:
            with torch.no_grad():
                eval_loss = 0
                for mbi, (input, label) in enumerate(testing_loader):
                    eval_loss0 = eval_loss_step(input, label)
                    eval_loss += eval_loss0 / len(testing_loader)

                eval_losses.append([epoch*len(training_loader), eval_loss.cpu()])
                print('Epoch {:d}: Validation loss : {:.04f}'.format(epoch, eval_loss.cpu()))
                plot_prediction(mod, u0 = args.u0, symms = symms, symm_method =
                                args.symm_method, eps = args.eps, plot_dir = args.pred_dir, epoch = epoch)

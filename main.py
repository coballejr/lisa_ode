import torch
import numpy as np
from args import Parser
from data.loaders import create_training_loader, create_eval_loader
from models import FullyConnected
from loss import PINNTrainingLoss, EvalLoss, EquLoss
from lisa.lie_field_torch import Scaling, Identity, Translation
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
        equ_test_symm = Identity()
    elif args.loss.lower() == 'lie_scale':
        if args.experiment.lower() == 'seperable':
            symms = (Identity(), Scaling())
            equ_test_symm = Scaling()

        else:
            raise NotImplementedError('Invalid experiment string.')

    elif args.loss.lower() == 'lie_trans':
        if args.experiment.lower() == 'seperable':
            symms = (Identity(), Translation())
            equ_test_symm = Translation()

        else:
            raise NotImplementedError('Invalid experiment string.')

    elif args.loss.lower() == 'lie_all':
        if args.experiment.lower() == 'seperable':
            symms = (Identity(), Translation(), Scaling())
            if args.test_symm_choice == 'translation':
                equ_test_symm = Translation()
            elif args.test_symm_choice == 'scaling':
                equ_test_symm = Scaling()
            else:
                raise NotImplementedError('Unknown test symm.')

        else:
            raise NotImplementedError('Invalid experiment string.')



    else:
        raise NotImplementedError('Invalid loss string.')

    # define losses
    training_loss_step = PINNTrainingLoss(mod,
                                          args.lambda_pde,
                                          args.lambda_init)
    eval_loss_step = EvalLoss(mod)
    equ_loss_step = EquLoss(mod, args.symm_method, equ_test_symm)

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
    eps_list = np.random.uniform(args.eps_range[0], args.eps_range[1], args.neps)
    iteration = 0
    train_losses = []
    eval_losses = []
    equ_losses = []
    for epoch in range(1, args.epochs+1):
        mod.train()
        training_loss = 0
        for mbi, (field_data, init_data) in enumerate(training_loader):
           optim.zero_grad()
           # forward
           for symm in symms:
               for eps in eps_list:
                    loss = training_loss_step(field_data, init_data, symm = symm,
                                              symm_method = args.symm_method, eps = eps)
                    training_loss += loss.detach() / (len(training_loader)*len(symms))

                    # backward
                    loss.backward()
                    optim.step()

        scheduler.step()
        train_losses.append([epoch*len(training_loader), training_loss.cpu()])

        # validation
        if epoch % args.test_freq == 0:
            with torch.no_grad():
                mod.eval()
                eval_loss = 0
                for mbi, (input, label) in enumerate(testing_loader):
                    eval_loss0 = eval_loss_step(input, label)
                    eval_loss += eval_loss0 / len(testing_loader)

                    # equivariance loss
                    eps_loss = 0
                    for eps in eps_list:
                        eps_loss += equ_loss_step(input, args.u0, eps) / (len(eps_list)*len(testing_loader))

                eval_losses.append([epoch*len(training_loader), eval_loss.cpu()])
                equ_losses.append([epoch*len(training_loader), eps_loss.cpu()])
                print('Epoch {:d}: Validation loss : {:.04f}'.format(epoch, eval_loss.cpu()))
                plot_prediction(mod, u0 = args.u0, symms = symms, symm_method =
                                args.symm_method, eps_list = eps_list, plot_dir = args.pred_dir, epoch = epoch)

         # ==== Checkpoint ====
        if epoch == 1 or epoch+1 % args.ckpt_freq == 0:
            file_name = "model_proc{:d}_{:d}.pt".format(0, epoch)
            torch.save(mod.state_dict(), args.ckpt_dir / file_name)

        # Save validation losses and print execution time
        np.save(args.run_dir / f"valid_{args.model}_{args.loss}", np.array(eval_losses))
        np.save(args.run_dir / f"train_{args.model}_{args.loss}", np.array(train_losses))
        np.save(args.run_dir / f"equiv_{args.model}_{args.loss}", np.array(equ_losses))

    print("Training complete.")


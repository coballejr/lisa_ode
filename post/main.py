from plot import Post

OUT_DIR = '../outputs'

if __name__ == '__main__':
    plotter = Post(OUT_DIR)

    # plot training
    training_losses = plotter.load('train')
    plotter.plot(training_losses, 'train.png')

    # plot validation
    validation_losses = plotter.load('valid')
    plotter.plot(validation_losses, 'valid.png')

    # plot equivariance
    equivariance_losses = plotter.load('equiv')
    plotter.plot(equivariance_losses, 'equiv.png')

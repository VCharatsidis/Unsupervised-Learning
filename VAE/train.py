import argparse
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from datasets.bmnist import bmnist
import matplotlib
from vae_core import VAE
import os


def main():
    data = bmnist()[:2]  # ignore test split

    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())
    size_width = 28
    max_loss = 1000

    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'encoder.model'
    encoder_model = os.path.join(script_directory, filepath)

    script_directory = os.path.split(os.path.abspath(__file__))[0]
    filepath = 'decoder.model'
    decoder_model = os.path.join(script_directory, filepath)
    min_loss = -1000

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)

        if min_loss < val_elbo:
            print("models saved iter: "+str(epoch))
            torch.save(model.encoder, encoder_model)
            torch.save(model.decoder, decoder_model)
            min_loss = val_elbo

        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")



    _, mean_sample = model.sample(64)
    save_sample(mean_sample, size_width, epoch)

    if ARGS.zdim == 2:
        print("manifold")
        manifold = model.manifold_sample(256)
        save_sample(manifold, size_width, epoch, 16)

    np.save('curves.npy', {'train': train_curve, 'val': val_curve})

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.
    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = 0
    size = len(data)

    for sample in data:
        input = sample.reshape(sample.shape[0], -1)

        elbo = model.forward(input)
        average_epoch_elbo -= elbo

        if model.training:
            model.zero_grad()
            elbo.backward()
            optimizer.step()

    average_epoch_elbo /= size

    return average_epoch_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(24, 12))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def save_sample(sample, size, epoch, nrow=8):
    sample = sample.view(-1, 1, size, size)
    sample = make_grid(sample, nrow=nrow).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"images/vae_manimani_{epoch}.png", sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=2, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()
    main()

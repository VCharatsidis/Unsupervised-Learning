import os
import torch
from datasets.bmnist import bmnist, my_bmnist
import torch
import matplotlib
from VAE.train import make_grid
import numpy as np
from VAE.vae_core import VAE
from torch.autograd import Variable


filepath = '..\VAE\\'
script_directory = os.path.split(os.path.abspath(__file__))[0]
encoder_model = os.path.join(script_directory, filepath + 'encoder.model')
encoder = torch.load(encoder_model)

decoder_model = os.path.join(script_directory, filepath + 'decoder.model')
decoder = torch.load(decoder_model)

data = my_bmnist('..\VAE\\data')[:2]

train, val = data

# for sample in train:
#     input = sample.reshape(sample.shape[0], -1)
#     print(input)


def display_reconstructions(number_sample):
    with torch.no_grad():
        x = train[number_sample].reshape(-1, 28 * 28)

        mean, std = encoder.forward(x)
        e = torch.zeros(mean.shape).normal_()
        sample = mean + std * e

        y = decoder(sample)
        im_means = y.reshape(1, 1, 28, 28)
        sampled_ims = torch.bernoulli(im_means)

        sample = sampled_ims.view(-1, 1, 28, 28)
        sample = make_grid(sample, nrow=1).detach().numpy().astype(np.float).transpose(1, 2, 0)
        matplotlib.image.imsave(f"images\mnist.png", sample)

        sample = train[number_sample].view(-1, 1, 28, 28)
        sample = make_grid(sample, nrow=1).detach().numpy().astype(np.float).transpose(1, 2, 0)
        matplotlib.image.imsave(f"images\original_{number_sample}.png", sample)


def display_centroid(vae_rep, number, gen_centroid):
    print(vae_rep)
    # mean = vae_rep[:2]
    # std = vae_rep[2:]
    #
    # e = torch.zeros(mean.shape).normal_()
    # sample = mean + std * e

    #sample = Variable(torch.FloatTensor(sample))

    vae_rep = Variable(torch.FloatTensor(vae_rep))
    y = decoder(vae_rep.float())
    im_means = y.reshape(1, 1, 28, 28)
    sampled_ims = torch.bernoulli(im_means)

    sample = sampled_ims.view(-1, 1, 28, 28)
    sample = make_grid(sample, nrow=1).detach().numpy().astype(np.float).transpose(1, 2, 0)
    matplotlib.image.imsave(f"images\\centroid_gen{gen_centroid}_{number}.png", sample)


def get_vae_representations():

    for sample in train:
        with torch.no_grad():
            x = sample.reshape(-1, 28 * 28)
            mean, std = encoder.forward(x)
            # mean = mean[0].detach().numpy()
            # std = std[0].detach().numpy()

            e = torch.zeros(mean.shape).normal_()
            sample = mean + std * e
            sample = sample[0].detach().numpy()

            file = open("vae_reps.txt", "a")
            for i in sample:
                file.writelines(str(i)+' ')

            file.writelines("\n")
            file.close()


get_vae_representations()
display_reconstructions(0)
display_reconstructions(1)
display_reconstructions(2)
display_reconstructions(3)
display_reconstructions(4)
display_reconstructions(5)
display_reconstructions(6)
display_reconstructions(7)
print(len(train))
print(len(val))

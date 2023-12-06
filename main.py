from datasets import ClassifDataset
from torch.backends import cudnn 
import torch
from torch.utils.data import DataLoader
import os
import argparse
from ddpm import DDPM
import matplotlib.pyplot as plt 
from nilearn import plotting
import numpy as np 
import nibabel as nib
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps


def train(config):

    if not os.path.isdir(config.sample_dir):
        os.mkdir(config.sample_dir)
    if not os.path.isdir(config.save_dir):
        os.mkdir(config.save_dir)

    ddpm = DDPM(config)

    # Data loader. 
    dataset_file = f'{config.data_dir}/{config.mode}-{config.dataset}.csv'

    dataset = ClassifDataset(
        dataset_file, 
        config.labels)

    print(f'Dataset {config.dataset}: \n {len(dataset)} images.')


    loader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        )

    optim = torch.optim.Adam(
        ddpm.parameters(), 
        lr=config.lrate
        )

    for ep in range(config.n_epoch):

        print(f'Epoch {ep}')

        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = config.lrate * (1 - ep / config.n_epoch)

        loss_ema = None

        for i, (x, c) in enumerate(loader):

            optim.zero_grad()

            x = x.to(ddpm.device)
            c = c.to(ddpm.device)

            loss = ddpm(x.float(), c.float())
            loss.backward()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()

            optim.step()

        print('Loss:', loss_ema)

        if ep%10==0:
            ddpm.eval()

            with torch.no_grad():

                n_sample = 1*config.n_classes

                for w_i, w in enumerate(config.ws_test):

                    x_gen, x_gen_store = ddpm.sample(
                        n_sample, 
                        (1, 48, 56, 48), 
                        guide_w=w
                        )

                    # append some real images at bottom, order by class also
                    x_real = torch.Tensor(x_gen.shape).to(ddpm.device)

                    for k in range(config.n_classes):
                        for j in range(int(n_sample/config.n_classes)):

                            try: 
                                idx = torch.squeeze((torch.argmax(c,dim=1) == k).nonzero())[j]

                            except:
                                idx = 0

                            x_real[k+(j*config.n_classes)] = x[idx]

                    x_all = torch.cat([x_gen, x_real])

                    fig,ax = plt.subplots(
                            nrows=2,
                            ncols=config.n_classes,
                            figsize=(config.n_classes*3, 10))

                    affine = np.array([[   4.,    0.,    0.,  -98.],
                                       [   0.,    4.,    0., -134.],
                                       [   0.,    0.,    4.,  -72.],
                                       [   0.,    0.,    0.,    1.]])

                    for n in range(n_sample):

                        img_xgen = nib.Nifti1Image(
                            np.array(
                                x_all[n].detach().cpu()
                                )[0,:,:,:], 
                            affine
                            )

                        img_xreal = nib.Nifti1Image(
                            np.array(
                                x_all[n_sample + n].detach().cpu()
                                )[0,:,:,:], 
                            affine
                            )

                        plotting.plot_glass_brain(
                            img_xgen, 
                            figure=fig, 
                            cmap=nilearn_cmaps['cold_hot'], 
                            plot_abs=False, 
                            title='Generated',
                            axes=ax[0, n],
                            display_mode = 'z')

                        plotting.plot_glass_brain(
                            img_xreal, 
                            figure=fig, 
                            cmap=nilearn_cmaps['cold_hot'], 
                            plot_abs=False, 
                            title='Real',
                            axes=ax[1, n],
                            display_mode = 'z')

                    plt.savefig(f'{config.sample_dir}/images_ep{ep}_w{w}.png')
                    plt.close()

            torch.save(ddpm.state_dict(), config.save_dir + f"/model_{ep}.pth")


def sample(config):
    ddpm = DDPM(config)
    ddpm.load_state_dict(
        torch.load(
            config.save_dir + f"/model_{config.test_iter}.pth", 
            map_location=ddpm.device
            )
        )

    # Data loader. 
    dataset_file = f'{config.data_dir}/valid-{config.dataset}.csv'

    dataset = ClassifDataset(
        dataset_file, 
        config.labels)

    print(f'Dataset {config.dataset}: \n {len(dataset)} images.')


    loader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        )

    x,c = next(iter(loader))

    ddpm.eval()

    with torch.no_grad():

        n_sample = 1*config.n_classes

        for w_i, w in enumerate(config.ws_test):

            x_gen, x_gen_store = ddpm.sample(
                n_sample, 
                (1, 48, 56, 48), 
                guide_w=w
                )

            # append some real images at bottom, order by class also
            x_real = torch.Tensor(x_gen.shape).to(ddpm.device)

            for k in range(config.n_classes):
                for j in range(int(n_sample/config.n_classes)):
                    x_real[k+(j*config.n_classes)] = x[k]

            x_all = torch.cat([x_gen, x_real])

            fig,ax = plt.subplots(
                    nrows=2,
                    ncols=config.n_classes,
                    figsize=(config.n_classes*2, 10))

            affine = np.array([[   4.,    0.,    0.,  -98.],
                                   [   0.,    4.,    0., -134.],
                                   [   0.,    0.,    4.,  -72.],
                                   [   0.,    0.,    0.,    1.]])

            for n in range(n_sample):

                img_xgen = nib.Nifti1Image(
                    np.array(
                        x_all[n].detach().cpu()
                        )[0,:,:,:], 
                    affine
                    )

                img_xreal = nib.Nifti1Image(
                    np.array(
                        x_all[n_sample + n].detach().cpu()
                        )[0,:,:,:], 
                    affine
                    )

                plotting.plot_glass_brain(
                    img_xgen, 
                    figure=fig, 
                    cmap=nilearn_cmaps['cold_hot'], 
                    plot_abs=False, 
                    title='Generated',
                    axes=ax[0, n],
                    display_mode = 'z')

                plotting.plot_glass_brain(
                    img_xreal, 
                    figure=fig, 
                    cmap=nilearn_cmaps['cold_hot'], 
                    plot_abs=False, 
                    title='Real',
                    axes=ax[1, n],
                    display_mode = 'z')

            plt.savefig(f'{config.sample_dir}/test-images_ep{config.test_iter}_w{w}.png')
            plt.close()

def transfer(config):
    ddpm = DDPM(config)
    ddpm.load_state_dict(
        torch.load(
            config.save_dir + f"/model_{config.test_iter}.pth", 
            map_location=ddpm.device
            )
        )

    # Data loader. 
    dataset_file = f'{config.data_dir}/valid-{config.dataset}.csv'

    dataset = ClassifDataset(
        dataset_file, 
        config.labels)

    print(f'Dataset {config.dataset}: \n {len(dataset)} images.')


    source_loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        )

    target_loader = DataLoader(
        dataset, 
        batch_size=config.n_classes, 
        shuffle=False, 
        )

    x,c = next(iter(source_loader))

    x_r,c_r = next(iter(target_loader))

    ddpm.eval()

    with torch.no_grad():

        for i in range(config.n_classes):

            c_t = torch.Tensor(c_r[i:i+1,:])

            for w_i, w in enumerate(config.ws_test):

                x_gen = ddpm.transfer(
                    x, 
                    c_t, 
                    guide_w=w
                    )

                fig,ax = plt.subplots(
                        nrows=3,
                        ncols=1,
                        figsize=(5, 10))

                affine = np.array([[   4.,    0.,    0.,  -98.],
                                   [   0.,    4.,    0., -134.],
                                   [   0.,    0.,    4.,  -72.],
                                   [   0.,    0.,    0.,    1.]])

                img_xgen = nib.Nifti1Image(
                    np.array(
                        x_gen.detach().cpu()
                        )[0,0,:,:,:], 
                    affine
                    )

                img_xreal = nib.Nifti1Image(
                    np.array(
                        x_r.detach().cpu()
                        )[i,0,:,:,:], 
                    affine
                    )

                img_xsrc = nib.Nifti1Image(
                    np.array(
                        x.detach().cpu()
                        )[0,0,:,:,:], 
                    affine
                    )

                plotting.plot_glass_brain(
                    img_xsrc, 
                    figure=fig, 
                    cmap=nilearn_cmaps['cold_hot'], 
                    plot_abs=False, 
                    title='Source',
                    axes=ax[0],
                    display_mode = 'z')

                plotting.plot_glass_brain(
                    img_xgen, 
                    figure=fig, 
                    cmap=nilearn_cmaps['cold_hot'], 
                    plot_abs=False, 
                    title='Generated',
                    axes=ax[1],
                    display_mode = 'z')

                plotting.plot_glass_brain(
                    img_xreal, 
                    figure=fig, 
                    cmap=nilearn_cmaps['cold_hot'], 
                    plot_abs=False, 
                    title='Target',
                    axes=ax[2],
                    display_mode = 'z')

                c_idx = torch.argmax(c, dim=1)
                c_t_idx = torch.argmax(c_t, dim=1)

                plt.savefig(f'{config.sample_dir}/test-images_ep{config.test_iter}_w{w}-orig_{c}-target_{c_t}.png')
                plt.close()

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='stargan/data')
    parser.add_argument('--dataset', type=str, default='global_dataset')
    parser.add_argument('--labels', type=str, help='conditions for generation',
                        default='pipelines')
    parser.add_argument('--sample_dir', type=str, default='sampling directory')
    parser.add_argument('--save_dir', type=str, default='save directory')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'transfer'])
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--n_epoch', type=int, default=500, help='number of total iterations')
    parser.add_argument('--lrate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--n_feat', type=int, default=64, help='number of features')
    parser.add_argument('--n_classes', type=int, default=24, help='number of classes')
    parser.add_argument('--beta', type=tuple, default=(1e-4, 0.02), help='number of classes')
    parser.add_argument('--n_T', type=int, default=500, help='number T')
    parser.add_argument('--drop_prob', type=float, default=0.1, help='probability drop')
    parser.add_argument('--ws_test', type=list, default=[0.0, 0.5, 2.0], help='weight strengh for sampling')
    parser.add_argument('--test_iter', type=int, default=10, help='epoch of model to test')

    config = parser.parse_args()

    if config.mode == 'train':
        train(config)

    elif config.mode == 'test':
        sample(config)

    elif config.mode == 'transfer':
        transfer(config)
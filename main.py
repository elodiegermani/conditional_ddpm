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
import pandas as pd
import nibabel as nib
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
import sys 
sys.path.insert(0, '../pipeline_classification/src')

def get_correlation(inim, outim):
        '''
        Compute the Pearson's correlation coefficient between original and reconstructed images.
        '''
        #orig, repro = utils.mask_using_original(inim, outim)
        
        data1 = inim.get_fdata().copy()
        data2 = outim.get_fdata().copy()
        
        # Vectorise input data
        data1 = np.reshape(data1, -1)
        data2 = np.reshape(data2, -1)

        in_mask_indices = np.logical_not(
            np.logical_or(
                np.logical_or(np.isnan(data1), np.absolute(data1) == 0),
                np.logical_or(np.isnan(data2), np.absolute(data2) == 0)))

        data1 = data1[in_mask_indices]
        data2 = data2[in_mask_indices]
        
        corr_coeff = np.corrcoef(data1, data2)[0][1]
        
        return corr_coeff

def class_change(model_param, image):
    package = 'lib.model'
    md = importlib.import_module(package)

    model = torch.load(model_param, map_location="cpu")
    
    if len(image.shape) == 4:
        image= image.unsqueeze(0)
        
    classe = torch.max(model(image.float().cpu()), 1)[1]

    return(classe)

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
                            title='Generated ',
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

    # x,c = next(iter(source_loader))

    # x_r,c_r = next(iter(target_loader))

    df_metrics = pd.DataFrame(
            columns = ['orig_label', 'target_label', 'orig-target', 'orig-gen', 'gen-target']
            )

    for n, (x, c) in enumerate(source_loader):

        ddpm.eval()

        with torch.no_grad():

            for i in range(config.n_classes):
                x_r, c_r = dataset[n//config.n_classes*config.n_classes+i]

                c_t = c_r.view(-1, c_r.shape[0])

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
                            )[0,:,:,:], 
                        affine
                        )

                    img_xsrc = nib.Nifti1Image(
                        np.array(
                            x.detach().cpu()
                            )[0,0,:,:,:], 
                        affine
                        )

                    if n % 50 == 0:

                        nib.save(img_xgen, f'{config.sample_dir}/gen-image_{n}-{config.dataset}_ep{config.test_iter}_w{w}-orig_{c_idx}-target_{c_t_idx}.nii.gz')
                        nib.save(img_xreal, f'{config.sample_dir}/trg-image_{n}-{config.dataset}_ep{config.test_iter}_w{w}-orig_{c_idx}-target_{c_t_idx}.nii.gz')
                        nib.save(img_xsrc, f'{config.sample_dir}/src-image_{n}-{config.dataset}_ep{config.test_iter}_w{w}-orig_{c_idx}-target_{c_t_idx}.nii.gz')

                    c_idx = torch.argmax(c, dim=1)[0]
                    c_t_idx = torch.argmax(c_t, dim=1)[0]

                    corr_orig_target = get_correlation(img_xsrc, img_xreal)
                    corr_orig_gen = get_correlation(img_xsrc, img_xgen)
                    corr_gen_target = get_correlation(img_xgen, img_xreal)

                    classe_orig = class_change(x)
                    classe_target = class_change(x_r)
                    classe_gen = class_change(x_gen)

                    df_img = pd.DataFrame({
                        'orig_label': [c_idx],
                        'target_label': [c_t_idx],
                        'orig-target': [corr_orig_target],
                        'orig-gen': [corr_orig_gen],
                        'gen-target': [corr_gen_target],
                        'gen_pred':[classe_gen],
                        'orig_pred':[classe_orig],
                        'target_pred':[classe_target]
                        })

                    df_metrics = pd.concat(
                        [df_metrics, df_img], 
                        ignore_index=True
                        ) 

                    print(df_metrics)

                    df_metrics.to_csv(f'{config.sample_dir}/df_metrics-{config.dataset}-w{w}.csv')

                    # if n%50==0:

                    #     plotting.plot_glass_brain(
                    #         img_xsrc, 
                    #         figure=fig, 
                    #         cmap=nilearn_cmaps['cold_hot'], 
                    #         plot_abs=False, 
                    #         title=f'Original, classe {dataset.label_list[c_idx]}',
                    #         axes=ax[0],
                    #         display_mode = 'z')

                    #     plotting.plot_glass_brain(
                    #         img_xgen, 
                    #         figure=fig, 
                    #         cmap=nilearn_cmaps['cold_hot'], 
                    #         plot_abs=False, 
                    #         title=f'Generated, classe {dataset.label_list[c_t_idx]}',
                    #         axes=ax[1],
                    #         display_mode = 'z')

                    #     plotting.plot_glass_brain(
                    #         img_xreal, 
                    #         figure=fig, 
                    #         cmap=nilearn_cmaps['cold_hot'], 
                    #         plot_abs=False, 
                    #         title=f'Target, classe {dataset.label_list[c_t_idx]}',
                    #         axes=ax[2],
                    #         display_mode = 'z')

                    #     plt.savefig(f'{config.sample_dir}/test-image_{n}-{config.dataset}_ep{config.test_iter}_w{w}-orig_{c_idx}-target_{c_t_idx}.png')
                    #     plt.close()

        
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
    parser.add_argument('--ws_test', type=list, default=[0.0, 2.0], help='weight strengh for sampling')
    parser.add_argument('--test_iter', type=int, default=10, help='epoch of model to test')

    config = parser.parse_args()

    if config.mode == 'train':
        train(config)

    elif config.mode == 'test':
        sample(config)

    elif config.mode == 'transfer':
        transfer(config)
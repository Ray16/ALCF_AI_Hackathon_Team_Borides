import os

import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as f

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions.multivariate_normal import MultivariateNormal

from tqdm import tqdm

from sklearn.metrics import r2_score


def train_model(vae_encoder, vae_decoder, property_network, x_train, x_test, y_train, y_test, num_epochs, batch_size,
                lr_enc, lr_dec, lr_property, KLD_alpha, sample_num, dtype, device, save_file_name, scaler_list):
    """
    Train the Variational Auto-Encoder
    """
    print('num_epochs: ', num_epochs, flush=True)

    # set optimizer
    optimizer_encoder = torch.optim.Adam(vae_encoder.parameters(), lr=lr_enc)
    optimizer_decoder = torch.optim.Adam(vae_decoder.parameters(), lr=lr_dec)
    optimizer_property_network = torch.optim.Adam(property_network.parameters(), lr=lr_property)

    # learning rate scheduler
    # lr_scheduler_encoder = ReduceLROnPlateau(optimizer_encoder, patience=30)
    # lr_scheduler_decoder = ReduceLROnPlateau(optimizer_decoder, patience=30)
    # lr_scheduler_property_network = ReduceLROnPlateau(optimizer_property_network, patience=30)

    num_batches_train = int(len(x_train) / batch_size)

    # scale target and feature
    y_train_scaled = torch.zeros(size=y_train.shape, dtype=dtype, device=device)
    y_test_scaled = torch.zeros(size=y_test.shape, dtype=dtype, device=device)

    for idx in range(y_train.shape[1]):
        y_train_scaled[:, idx] = torch.tensor(
            scaler_list[idx].transform(y_train[:, idx].view(-1, 1).cpu()).flatten(),
            dtype=dtype, device=device
        )
        y_test_scaled[:, idx] = torch.tensor(
            scaler_list[idx].transform(y_test[:, idx].view(-1, 1).cpu()).flatten(),
            dtype=dtype, device=device
        )

    # training
    for epoch in range(num_epochs):
        tqdm.write("===== EPOCH %d =====" % (epoch + 1))
        # random permutation
        start = time.time()

        # train mode
        vae_encoder.train()
        vae_decoder.train()
        property_network.train()

        # mini-batch training
        tqdm.write("Training ...")
        pbar = tqdm(range(num_batches_train), total=num_batches_train, leave=True)
        time.sleep(2.0)
        with pbar as t:
            for batch_iteration in t:
                # manual batch iterations
                start_idx = batch_iteration * batch_size
                stop_idx = (batch_iteration + 1) * batch_size

                if batch_iteration == num_batches_train - 1:
                    stop_idx = x_train.shape[0] - 1

                batch_data = x_train[start_idx: stop_idx]
                y_batch_train = y_train_scaled[start_idx: stop_idx]
                inp_flat_one_hot = batch_data.flatten(start_dim=1)  # [b, 32*2]
                latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)  # [b, latent_dimension]

                # initialization hidden internal state of RNN (RNN has two inputs and two outputs)
                #    input: latent space & hidden state
                #    output: one-hot encoding of one character of molecule & hidden
                #    state the hidden state acts as the internal memory
                latent_points = latent_points.unsqueeze(0)  # [1, b, latent_dimension]
                hidden = vae_decoder.init_hidden(batch_size=latent_points.shape[1])  # [1, b, 12]

                # decode
                out_one_hot = torch.zeros_like(batch_data, dtype=dtype, device=device)  # [b, 32, 2]
                for seq_index in range(batch_data.shape[1]):  # 32 (total beads)
                    out_one_hot_line, hidden = vae_decoder(latent_points, hidden)
                    out_one_hot[:, seq_index, :] = out_one_hot_line[0]

                # compute ELBO
                vae_loss = compute_elbo(batch_data, out_one_hot, mus, log_vars, KLD_alpha)

                # compute reconloss + WAE
                # vae_loss = compute_elbo_with_mmd(batch_data, out_one_hot, latent_points[0])

                # compute property loss
                y_batch_train_feature = y_batch_train[:, :-1]
                latent_points_with_features = torch.cat([latent_points[0], y_batch_train_feature], dim=-1)
                property_prediction = property_network(latent_points_with_features)

                y_batch_train_target = y_batch_train[:, -1].view(-1, 1)
                property_loss = nn.MSELoss()(property_prediction, y_batch_train_target)

                # total loss
                loss = vae_loss + property_loss

                # perform back propogation
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
                optimizer_property_network.zero_grad()

                loss.backward(retain_graph=True)

                nn.utils.clip_grad_norm_(vae_encoder.parameters(), 0.05)
                nn.utils.clip_grad_norm_(vae_decoder.parameters(), 0.05)
                nn.utils.clip_grad_norm_(property_network.parameters(), 0.05)

                optimizer_encoder.step()
                optimizer_decoder.step()
                optimizer_property_network.step()

                # lr_scheduler_encoder.step(float(loss))
                # lr_scheduler_decoder.step(float(loss))
                # lr_scheduler_property_network.step(float(loss))

                # update progress bar
                t.set_postfix(vae_loss='%.2f' % float(vae_loss), property_loss='%.2f' % float(property_loss))
                t.update()
            t.close()

        tqdm.write("Validation ...")
        # validation
        with torch.no_grad():
            # eval mode
            vae_encoder.eval()
            vae_decoder.eval()
            property_network.eval()

            # data
            inp_flat_one_hot = x_test.flatten(start_dim=1)  # [b, 32*2]
            latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)  # [b, latent_dimension]
            latent_points = latent_points.unsqueeze(0)  # [1, b, 32*2]
            hidden = vae_decoder.init_hidden(batch_size=latent_points.shape[1])  # [1, b, 12]

            # decode
            out_one_hot = torch.zeros_like(x_test, dtype=dtype, device=device)  # [b, 32, 2]
            for seq_index in range(x_test.shape[1]):  # 32 (total beads)
                out_one_hot_line, hidden = vae_decoder(latent_points, hidden)
                out_one_hot[:, seq_index, :] = out_one_hot_line[0]

            # compute recon quality - component level
            element_quality, sequence_quality = compute_recon_quality(x_test, out_one_hot)

            # compute ELBO
            vae_loss = compute_elbo(x_test, out_one_hot, mus, log_vars, KLD_alpha)

            # compute wasserstein loss
            # vae_loss = compute_elbo_with_mmd(x_test, out_one_hot, latent_points[0])

            # compute property loss
            y_test_feature = y_test_scaled[:, :-1]
            latent_points_with_features = torch.cat([latent_points[0], y_test_feature], dim=-1)
            property_prediction = property_network(latent_points_with_features)

            y_test_target = y_test_scaled[:, -1].view(-1, 1)
            property_loss = nn.MSELoss()(property_prediction, y_test_target)

            # total loss
            loss = vae_loss + property_loss
            tqdm.write("Loss is %.3f" % float(loss))
            tqdm.write("Elementary quality score is %.3f / 100" % float(element_quality))
            tqdm.write("Sequence quality score is %.3f / 100" % float(sequence_quality))

        end = time.time()
        tqdm.write("EPOCH %d takes %.3f minutes" % ((epoch + 1), (end - start) / 60))

    # MSE prediction
    with torch.no_grad():
        # eval mode
        vae_encoder.eval()
        vae_decoder.eval()
        property_network.eval()

        # calculate final decoding score
        x_total = torch.cat([x_train, x_test], dim=0)
        total_inp_flat_one_hot = x_total.flatten(start_dim=1)  # [b, 32*2]
        total_latent_points, mus, log_vars = vae_encoder(total_inp_flat_one_hot)  # [b, latent_dimension]
        total_latent_points = total_latent_points.unsqueeze(0)  # [1, b, 32*2]
        hidden = vae_decoder.init_hidden(batch_size=total_latent_points.shape[1])  # [1, b, 12]

        # decode
        out_one_hot = torch.zeros_like(x_total, dtype=dtype, device=device)  # [b, 32, 2]
        for seq_index in range(x_total.shape[1]):  # 32 (total beads)
            out_one_hot_line, hidden = vae_decoder(total_latent_points, hidden)
            out_one_hot[:, seq_index, :] = out_one_hot_line[0]

        element_quality, sequence_quality = compute_recon_quality(x_total, out_one_hot)
        tqdm.write("Final elementary quality score is %.3f / 100" % float(element_quality))
        tqdm.write("Final S=sequence quality score is %.3f / 100" % float(sequence_quality))

        # calculate r2 score
        train_inp_flat_one_hot = x_train.flatten(start_dim=1)  # [b, 32*2]
        train_latent_points, mus, log_vars = vae_encoder(train_inp_flat_one_hot)  # [b, latent_dimension]
        train_latent_points_with_feature = torch.cat([train_latent_points, y_train_scaled[:, :-1]], dim=-1)

        test_inp_flat_one_hot = x_test.flatten(start_dim=1)  # [b, 32*2]
        test_latent_points, mus, log_vars = vae_encoder(test_inp_flat_one_hot)  # [b, latent_dimension]
        test_latent_points_with_feature = torch.cat([test_latent_points, y_test_scaled[:, :-1]], dim=-1)

        pred_train_final = scaler_list[-1].inverse_transform(property_network(train_latent_points_with_feature).cpu())
        pred_test_final = scaler_list[-1].inverse_transform(property_network(test_latent_points_with_feature).cpu())

        # get r2 score
        train_r2_score = r2_score(y_true=y_train[:, -1].view(-1, 1).cpu(), y_pred=pred_train_final)
        test_r2_score = r2_score(y_true=y_test[:, -1].view(-1, 1).cpu(), y_pred=pred_test_final)

        plt.plot(pred_train_final, y_train[:, -1].view(-1, 1).cpu(), 'bo', label='Train R2 score: %.3f' % train_r2_score)
        plt.plot(pred_test_final, y_test[:, -1].view(-1, 1).cpu(), 'ro', label='Test R2 score: %.3f' % test_r2_score)

        plt.legend()
        plt.xlabel('prediction')
        plt.ylabel('target')

        save_directory = os.path.join(os.getcwd(), save_file_name)
        plt.savefig(save_directory)
        plt.show()
        plt.close()

    return vae_encoder, vae_decoder, property_network


def compute_elbo(x, x_hat, mus, log_vars, KLD_alpha):
    inp = x_hat.reshape(-1, x_hat.shape[2])
    target = x.reshape(-1, x.shape[2]).argmax(1)

    criterion = torch.nn.CrossEntropyLoss()
    recon_loss = criterion(inp, target)
    kld = -0.5 * torch.mean(1. + log_vars - mus.pow(2) - log_vars.exp())

    return recon_loss + KLD_alpha * kld


def compute_elbo_with_mmd(x, x_hat, posterior_z_samples):
    inp = x_hat.reshape(-1, x_hat.shape[2])
    target = x.reshape(-1, x.shape[2]).argmax(1)

    criterion = torch.nn.CrossEntropyLoss()
    recon_loss = criterion(inp, target)

    # set variables
    device = inp.device
    dtype = inp.dtype
    latent_dimension = posterior_z_samples.shape[1]
    number_of_samples = posterior_z_samples.shape[0]

    # get prior samples
    gaussian_sampler = MultivariateNormal(
        loc=torch.zeros(latent_dimension, dtype=dtype, device=device),
        covariance_matrix=torch.eye(n=latent_dimension, dtype=dtype, device=device)
    )
    prior_z_samples = gaussian_sampler.sample(sample_shape=(number_of_samples,))

    # calculate Maximum Mean Discrepancy with inverse multi-quadratics kernel
    # set value of c - refer to Sec.4 of Wasserstein paper
    c = 2 * latent_dimension * (1.0 ** 2)

    # calculate pp term (p means prior)
    pp = torch.mm(prior_z_samples, prior_z_samples.t())
    pp_diag = pp.diag().unsqueeze(0).expand_as(pp)
    kernel_pp = c / (c + pp_diag + pp_diag.t() - 2 * pp)
    kernel_pp = (torch.sum(kernel_pp) - number_of_samples) / (number_of_samples * (number_of_samples - 1))

    # calculate qq term (q means posterior)
    qq = torch.mm(posterior_z_samples, posterior_z_samples.t())
    qq_diag = qq.diag().unsqueeze(0).expand_as(qq)
    kernel_qq = c / (c + qq_diag + qq_diag.t() - 2 * qq)
    kernel_qq = (torch.sum(kernel_qq) - number_of_samples) / (number_of_samples * (number_of_samples - 1))

    # calculate pq term
    pq = pp_diag.t() - torch.mm(prior_z_samples, posterior_z_samples.t()) \
        - torch.mm(posterior_z_samples, prior_z_samples.t()) + qq_diag
    kernel_pq = -2 * torch.sum(c / (c + pq)) / number_of_samples ** 2

    mmd = kernel_pp + kernel_qq + kernel_pq

    return recon_loss + mmd


def compute_recon_quality(x, x_hat):
    x_indices = x.reshape(-1, x.shape[2]).argmax(1)
    x_hat_prob = f.softmax(x_hat, dim=-1)
    x_hat_indices = x_hat_prob.reshape(-1, x_hat_prob.shape[2]).argmax(1)

    # calculate elementary level quality
    differences = 1. - torch.abs(x_hat_indices - x_indices)
    differences = torch.clamp(differences, min=0., max=1.).double()
    element_quality = 100. * torch.mean(differences)
    element_quality = element_quality.detach().cpu().numpy()

    # calculate sequence level quality
    x_sequence_indices = x_indices.reshape(x.shape[0], -1)
    x_hat_sequence_indices = x_hat_indices.reshape(x_hat.shape[0], -1)
    sequence_difference = x_hat_sequence_indices - x_sequence_indices
    sequence_difference = 1. - torch.abs(sequence_difference.sum(dim=-1))
    sequence_difference = torch.clamp(sequence_difference, min=0., max=1.)
    sequence_quality = 100. * torch.mean(sequence_difference)
    sequence_quality = sequence_quality.detach().cpu().numpy()

    return element_quality, sequence_quality


# def quality_in_valid_set(vae_encoder, vae_decoder, data_valid, batch_size):
#     device = data_valid.device
#     data_valid = data_valid[torch.randperm(data_valid.size()[0])]  # shuffle
#     num_batches_valid = len(data_valid) // batch_size
#
#     quality_list = []
#     for batch_iteration in range(min(25, num_batches_valid)):
#
#         # get batch
#         start_idx = batch_iteration * batch_size
#         stop_idx = (batch_iteration + 1) * batch_size
#         batch = data_valid[start_idx: stop_idx]
#         _, trg_len, _ = batch.size()
#
#         inp_flat_one_hot = batch.flatten(start_dim=1)
#         latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)
#
#         latent_points = latent_points.unsqueeze(0)
#         hidden = vae_decoder.init_hidden(batch_size=batch_size)
#         out_one_hot = torch.zeros_like(batch, device=device)
#         for seq_index in range(trg_len):
#             out_one_hot_line, hidden = vae_decoder(latent_points, hidden)
#             out_one_hot[:, seq_index, :] = out_one_hot_line[0]
#
#         # assess reconstruction quality
#         quality = compute_recon_quality(batch, out_one_hot)
#         quality_list.append(quality)
#
#     return np.mean(quality_list).item()


# def latent_space_quality(vae_encoder, vae_decoder, type_of_encoding,
#                          alphabet, sample_num, sample_len):
#     total_correct = 0
#     all_correct_molecules = set()
#     print(f"latent_space_quality:"
#           f" Take {sample_num} samples from the latent space")
#
#     for _ in range(1, sample_num + 1):
#
#         molecule_pre = ''
#         for i in sample_latent_space(vae_encoder, vae_decoder, sample_len):
#             molecule_pre += alphabet[i]
#         molecule = molecule_pre.replace(' ', '')
#
#         # if type_of_encoding == 1:  # if SELFIES, decode to SMILES
#         #     molecule = sf.decoder(molecule)
#
#         # if is_correct_smiles(molecule):
#         #     total_correct += 1
#         #     all_correct_molecules.add(molecule)
#
#     return total_correct, len(all_correct_molecules)
#
#
# def sample_latent_space(vae_encoder, vae_decoder, sample_len):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     vae_encoder.eval()
#     vae_decoder.eval()
#
#     gathered_atoms = []
#
#     fancy_latent_point = torch.randn(1, 1, vae_encoder.latent_dimension, device=device)
#     hidden = vae_decoder.init_hidden()
#
#     # runs over letters from molecules (len=size of largest molecule)
#     for _ in range(sample_len):
#         out_one_hot, hidden = vae_decoder(fancy_latent_point, hidden)
#
#         out_one_hot = out_one_hot.flatten().detach()
#         soft = nn.Softmax(0)
#         out_one_hot = soft(out_one_hot)
#
#         out_index = out_one_hot.argmax(0)
#         gathered_atoms.append(out_index.data.cpu().tolist())
#
#     vae_encoder.train()
#     vae_decoder.train()
#
#     return gathered_atoms

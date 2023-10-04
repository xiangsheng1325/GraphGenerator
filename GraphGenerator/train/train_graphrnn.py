import os
import sys

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.decomposition import PCA
import logging
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from time import gmtime, strftime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from random import shuffle
import pickle
from tensorboard_logger import configure, log_value
import scipy.misc
import time as tm

from GraphGenerator.utils.train_utils import *
from GraphGenerator.models.graphrnn import *
from GraphGenerator.utils.data_utils import *
from GraphGenerator.metrics.memory import get_peak_gpu_memory
# from args import Args
# import create_graphs


def train_vae_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].astype(float)
        y_unsorted = data['y'].astype(float)
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0), device=args.device)

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x).to(args.device)
        y = Variable(y).to(args.device)

        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        y_pred,z_mu,z_lsgms = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        z_mu = pack_padded_sequence(z_mu, y_len, batch_first=True)
        z_mu = pad_packed_sequence(z_mu, batch_first=True)[0]
        z_lsgms = pack_padded_sequence(z_lsgms, y_len, batch_first=True)
        z_lsgms = pad_packed_sequence(z_lsgms, batch_first=True)[0]
        # use cross entropy loss
        loss_bce = binary_cross_entropy_weight(y_pred, y)
        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl /= y.size(0)*y.size(1)*sum(y_len) # normalize
        loss = loss_bce + loss_kl
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        z_mu_mean = torch.mean(z_mu.data)
        z_sgm_mean = torch.mean(z_lsgms.mul(0.5).exp_().data)
        z_mu_min = torch.min(z_mu.data)
        z_sgm_min = torch.min(z_lsgms.mul(0.5).exp_().data)
        z_mu_max = torch.max(z_mu.data)
        z_sgm_max = torch.max(z_lsgms.mul(0.5).exp_().data)


        if epoch % args.train.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train bce loss: {:.6f}, train kl loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.train.epochs,loss_bce.data[0], loss_kl.data[0], args.dataset.name, args.model.num_layers, args.model.hidden_size_rnn))
            print('z_mu_mean', z_mu_mean, 'z_mu_min', z_mu_min, 'z_mu_max', z_mu_max, 'z_sgm_mean', z_sgm_mean, 'z_sgm_min', z_sgm_min, 'z_sgm_max', z_sgm_max)

        # logging
        log_value('bce_loss_'+args.fname, loss_bce.data[0], epoch*args.batch_ratio+batch_idx)
        log_value('kl_loss_' +args.fname, loss_kl.data[0], epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_mean_'+args.fname, z_mu_mean, epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_min_'+args.fname, z_mu_min, epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_max_'+args.fname, z_mu_max, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_mean_'+args.fname, z_sgm_mean, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_min_'+args.fname, z_sgm_min, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_max_'+args.fname, z_sgm_max, epoch*args.batch_ratio + batch_idx)

        loss_sum += loss.data
    return loss_sum/(batch_idx+1)

def test_vae_epoch(epoch, args, rnn, output, test_batch_size=16, save_histogram=False, sample_time = 1):
    rnn.hidden = rnn.init_hidden(test_batch_size, device=args.device)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.model.max_num_node)
    y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.model.max_prev_node)).to(args.device) # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.model.max_prev_node)).to(args.device) # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.model.max_prev_node)).to(args.device)
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step, _, _ = output(h)
        y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).to(args.device)
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    # save prediction histograms, plot histogram over each time step
    # if save_histogram:
    #     save_prediction_histogram(y_pred_data.cpu().numpy(),
    #                           fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
    #                           max_num_node=max_num_node)


    return G_pred_list


def test_vae_partial_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].astype(float)
        y = data['y'].astype(float)
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size, device=args.device)
        # generate graphs
        max_num_node = int(args.model.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.model.max_prev_node)).to(args.device) # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.model.max_prev_node)).to(args.device) # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,args.model.max_prev_node)).to(args.device)
        for i in range(max_num_node):
            print('finish node',i)
            h = rnn(x_step)
            y_pred_step, _, _ = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:].to(args.device), current=i, y_len=y_len, sample_time=sample_time)

            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).to(args.device)
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list



def train_mlp_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        # x_unsorted = data['x'].astype(float)
        x_unsorted = torch.from_numpy(data['x']).float().unsqueeze(0)
        # y_unsorted = data['y'].astype(float)
        y_unsorted = torch.from_numpy(data['y']).float().unsqueeze(0)
        y_len_unsorted = data['len']
        if isinstance(y_len_unsorted, int):
            y_len_unsorted = torch.tensor([y_len_unsorted], dtype=torch.long)
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0), device=args.device)

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x).to(args.device)
        y = Variable(y).to(args.device)

        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        if epoch % args.train.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}, memory: {} MiB'.format(
                epoch, args.train.epochs, loss.data, args.dataset.name, args.model.num_layers,
                args.model.hidden_size_rnn, get_peak_gpu_memory(args.device) // 1024 // 1024))

        # logging
        # log_value('loss_'+args.fname, loss.data, epoch*args.batch_ratio+batch_idx)

        loss_sum += loss.data
    return loss_sum/(batch_idx+1)


def test_mlp_epoch(epoch, args, rnn, output, test_batch_size=16, save_histogram=False,sample_time=1):
    rnn.hidden = rnn.init_hidden(test_batch_size, device=args.device)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.model.max_num_node)
    y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.model.max_prev_node)).to(args.device) # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.model.max_prev_node)).to(args.device) # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.model.max_prev_node)).to(args.device)
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step = output(h)
        y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).to(args.device)
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)


    # # save prediction histograms, plot histogram over each time step
    # if save_histogram:
    #     save_prediction_histogram(y_pred_data.cpu().numpy(),
    #                           fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
    #                           max_num_node=max_num_node)
    return G_pred_list



def test_mlp_partial_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].astype(float)
        y = data['y'].astype(float)
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size, device=args.device)
        # generate graphs
        max_num_node = int(args.model.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.model.max_prev_node)).to(args.device) # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.model.max_prev_node)).to(args.device) # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,args.model.max_prev_node)).to(args.device)
        for i in range(max_num_node):
            print('finish node',i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:].to(args.device), current=i, y_len=y_len, sample_time=sample_time)

            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).to(args.device)
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list


def test_mlp_partial_simple_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].astype(float)
        y = data['y'].astype(float)
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size, device=args.device)
        # generate graphs
        max_num_node = int(args.model.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.model.max_prev_node)).to(args.device) # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.model.max_prev_node)).to(args.device) # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,args.model.max_prev_node)).to(args.device)
        for i in range(max_num_node):
            print('finish node',i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised_simple(y_pred_step, y[:,i:i+1,:].to(args.device), current=i, y_len=y_len, sample_time=sample_time)

            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).to(args.device)
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list


def train_mlp_forward_epoch(epoch, args, rnn, output, data_loader):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].astype(float)
        y_unsorted = data['y'].astype(float)
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0), device=args.device)

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x).to(args.device)
        y = Variable(y).to(args.device)

        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss

        loss = 0
        for j in range(y.size(1)):
            # print('y_pred',y_pred[0,j,:],'y',y[0,j,:])
            end_idx = min(j+1,y.size(2))
            loss += binary_cross_entropy_weight(y_pred[:,j,0:end_idx], y[:,j,0:end_idx])*end_idx


        if epoch % args.train.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.train.epochs,loss.data, args.dataset.name, args.model.num_layers, args.model.hidden_size_rnn))

        # logging
        log_value('loss_'+args.fname, loss.data, epoch*args.batch_ratio+batch_idx)

        loss_sum += loss.data
    return loss_sum/(batch_idx+1)





## too complicated, deprecated
# def test_mlp_partial_bfs_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
#     rnn.eval()
#     output.eval()
#     G_pred_list = []
#     for batch_idx, data in enumerate(data_loader):
#         x = data['x'].astype(float
#         y = data['y'].astype(float
#         y_len = data['len']
#         test_batch_size = x.size(0)
#         rnn.hidden = rnn.init_hidden(test_batch_size)
#         # generate graphs
#         max_num_node = int(args.model.max_num_node)
#         y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.model.max_prev_node)).to(args.device # normalized prediction score
#         y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.model.max_prev_node)).to(args.device # discrete prediction
#         x_step = Variable(torch.ones(test_batch_size,1,args.model.max_prev_node)).to(args.device
#         for i in range(max_num_node):
#             # 1 back up hidden state
#             hidden_prev = Variable(rnn.hidden.data).to(args.device
#             h = rnn(x_step)
#             y_pred_step = output(h)
#             y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
#             x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:].to(args.device, current=i, y_len=y_len, sample_time=sample_time)
#             y_pred_long[:, i:i + 1, :] = x_step
#
#             rnn.hidden = Variable(rnn.hidden.data).to(args.device
#
#             print('finish node', i)
#         y_pred_data = y_pred.data
#         y_pred_long_data = y_pred_long.data.long()
#
#         # save graphs as pickle
#         for i in range(test_batch_size):
#             adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
#             G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
#             G_pred_list.append(G_pred)
#     return G_pred_list


def train_rnn_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        # x_unsorted = data['x'].astype(float)
        x_unsorted = torch.from_numpy(data['x']).float().unsqueeze(0)
        # y_unsorted = data['y'].astype(float)
        y_unsorted = torch.from_numpy(data['y']).float().unsqueeze(0)
        y_len_unsorted = data['len']
        if isinstance(y_len_unsorted, int):
            y_len_unsorted = torch.tensor([y_len_unsorted], dtype=torch.long)
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0), device=args.device)
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x).to(args.device)
        y = Variable(y).to(args.device)
        output_x = Variable(output_x).to(args.device)
        output_y = Variable(output_y).to(args.device)
        # print(output_y_len)
        # print('len',len(output_y_len))
        # print('y',y.size())
        # print('output_y',output_y.size())


        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).to(args.device)
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(args.model.num_layers-1, h.size(0), h.size(1))).to(args.device)
        output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, output_y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        if epoch % args.train.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}, memory: {} MiB'.format(
                epoch, args.train.epochs,loss.data, args.dataset.name, args.model.num_layers,
                args.model.hidden_size_rnn, get_peak_gpu_memory(args.device)//1024//1024))

        # logging
        # log_value('loss_'+args.fname, loss.data, epoch*args.batch_ratio+batch_idx)
        # log_value('loss_' + args.fname, loss.data, epoch * args.batch_ratio + batch_idx)
        feature_dim = y.size(1)*y.size(2)
        # loss_sum += loss.data*feature_dim
        loss_sum += loss.data * feature_dim
    return loss_sum/(batch_idx+1)



def test_rnn_epoch(epoch, args, rnn, output, test_batch_size=16):
    rnn.hidden = rnn.init_hidden(test_batch_size, device=args.device)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.model.max_num_node)
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.model.max_prev_node)).to(args.device) # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.model.max_prev_node)).to(args.device)
    for i in range(max_num_node):
        h = rnn(x_step)
        # output.hidden = h.permute(1,0,2)
        hidden_null = Variable(torch.zeros(args.model.num_layers - 1, h.size(0), h.size(2))).to(args.device)
        output.hidden = torch.cat((h.permute(1,0,2), hidden_null),
                                  dim=0)  # num_layers, batch_size, hidden_size
        x_step = Variable(torch.zeros(test_batch_size,1,args.model.max_prev_node)).to(args.device)
        output_x_step = Variable(torch.ones(test_batch_size,1,1)).to(args.device)
        for j in range(min(args.model.max_prev_node,i+1)):
            output_y_pred_step = output(output_x_step)
            output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
            x_step[:,:,j:j+1] = output_x_step
            output.hidden = Variable(output.hidden.data).to(args.device)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).to(args.device)
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list




def train_rnn_forward_epoch(epoch, args, rnn, output, data_loader):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].astype(float)
        y_unsorted = data['y'].astype(float)
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0), device=args.device)
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x).to(args.device)
        y = Variable(y).to(args.device)
        output_x = Variable(output_x).to(args.device)
        output_y = Variable(output_y).to(args.device)
        # print(output_y_len)
        # print('len',len(output_y_len))
        # print('y',y.size())
        # print('output_y',output_y.size())


        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).to(args.device)
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(args.model.num_layers-1, h.size(0), h.size(1))).to(args.device)
        output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, output_y)


        if epoch % args.train.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.train.epochs,loss.data, args.dataset.name, args.model.num_layers, args.model.hidden_size_rnn))

        # logging
        log_value('loss_'+args.fname, loss.data, epoch*args.batch_ratio+batch_idx)
        # print(y_pred.size())
        feature_dim = y_pred.size(0)*y_pred.size(1)
        loss_sum += loss.data*feature_dim/y.size(0)
    return loss_sum/(batch_idx+1)


########### train function for LSTM + VAE
def train(args, dataset_train, rnn, output):
    # check if load existing model
    if args.load:
        fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
        rnn.load_state_dict(torch.load(fname))
        fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
        output.load_state_dict(torch.load(fname))

        # args.lr = 0.00001
        args.lr = 0.003
        epoch = args.load_epoch
        print('model loaded!, lr: {}'.format(args.lr))
    else:
        epoch = 1

    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)

    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.train.epochs)
    while epoch<=args.train.epochs:
        time_start = tm.time()
        print("epoch {} ({})".format(epoch,time_start))
        # train
        if 'GraphRNN_VAE' in args.model.name:
            train_vae_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_MLP' in args.model.name:
            train_mlp_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_RNN' in args.model.name:
            train_rnn_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        if epoch % args.train.epochs_test == 0 and epoch>=args.train.epochs_test_start:
            for sample_time in range(1,4):
                G_pred = []
                while len(G_pred)<args.test_total_size:
                    if 'GraphRNN_VAE' in args.model.name:
                        tmpname = 'graphrnn-vae'
                        G_pred_step = test_vae_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    elif 'GraphRNN_MLP' in args.model.name:
                        tmpname = 'graphrnn-mlp'
                        G_pred_step = test_mlp_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    elif 'GraphRNN_RNN' in args.model.name:
                        tmpname = 'graphrnn-rnn'
                        G_pred_step = test_rnn_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size)
                    G_pred.extend(G_pred_step)
                # save graphs
                # general experiment
                # fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + '.dat'
                # save_graph_list(G_pred, fname)
                # tuning experiment
                fname = '/home/xiangsheng/venv/ggen/ggen/generators/result/tuning/rnn'+ tmpname[-3:] +'-emb/'\
                        + args.dataset.name + '_to_' + tmpname + '_' + str(sample_time)\
                        + '_emb' + str(args.model.hidden_size_rnn) + '.dat'
                save_graph_list(G_pred, fname)
                if 'GraphRNN_RNN' in args.model.name:
                    break
            print('test done, graphs saved')


        # save model checkpoint
        if args.save:
            if epoch % args.train.epochs_save == 0:
                fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
                torch.save(rnn.state_dict(), fname)
                fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
                torch.save(output.state_dict(), fname)
        epoch += 1
    np.save(args.timing_save_path+args.fname,time_all)


########### for graph completion task
def train_graph_completion(args, dataset_test, rnn, output):
    fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
    rnn.load_state_dict(torch.load(fname))
    fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
    output.load_state_dict(torch.load(fname))

    epoch = args.load_epoch
    print('model loaded!, epoch: {}'.format(args.load_epoch))

    for sample_time in range(1,4):
        if 'GraphRNN_MLP' in args.model.name:
            G_pred = test_mlp_partial_simple_epoch(epoch, args, rnn, output, dataset_test,sample_time=sample_time)
        if 'GraphRNN_VAE' in args.model.name:
            G_pred = test_vae_partial_epoch(epoch, args, rnn, output, dataset_test,sample_time=sample_time)
        # save graphs
        fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + 'graph_completion.dat'
        save_graph_list(G_pred, fname)
    print('graph completion done, graphs saved')


########### for NLL evaluation
def train_nll(args, dataset_train, dataset_test, rnn, output,graph_validate_len,graph_test_len, max_iter = 1000):
    fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
    rnn.load_state_dict(torch.load(fname))
    fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
    output.load_state_dict(torch.load(fname))

    epoch = args.load_epoch
    print('model loaded!, epoch: {}'.format(args.load_epoch))
    fname_output = args.nll_save_path + args.model.name + '_' + args.dataset.name + '.csv'
    with open(fname_output, 'w+') as f:
        f.write(str(graph_validate_len)+','+str(graph_test_len)+'\n')
        f.write('train,test\n')
        for iter in range(max_iter):
            if 'GraphRNN_MLP' in args.model.name:
                nll_train = train_mlp_forward_epoch(epoch, args, rnn, output, dataset_train)
                nll_test = train_mlp_forward_epoch(epoch, args, rnn, output, dataset_test)
            if 'GraphRNN_RNN' in args.model.name:
                nll_train = train_rnn_forward_epoch(epoch, args, rnn, output, dataset_train)
                nll_test = train_rnn_forward_epoch(epoch, args, rnn, output, dataset_test)
            print('train',nll_train,'test',nll_test)
            f.write(str(nll_train)+','+str(nll_test)+'\n')

    print('NLL evaluation done')


def main_train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('CUDA', args.cuda)
    print('File name prefix', args.fname)
    # check if necessary directories exist
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)
    if not os.path.isdir(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.isdir(args.timing_save_path):
        os.makedirs(args.timing_save_path)
    if not os.path.isdir(args.figure_prediction_save_path):
        os.makedirs(args.figure_prediction_save_path)
    if not os.path.isdir(args.nll_save_path):
        os.makedirs(args.nll_save_path)

    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    # logging.basicConfig(filename='logs/train' + time + '.log', level=logging.DEBUG)
    if args.clean_tensorboard:
        if os.path.isdir("tensorboard"):
            shutil.rmtree("tensorboard")
    configure("tensorboard/run" + time, flush_secs=5)

    graphs = create(args)

    # split datasets
    random.seed(123)
    shuffle(graphs)
    graphs_len = len(graphs)
    graphs_test = graphs[int(0.8 * graphs_len):]
    # graphs_train = graphs[0:int(0.8*graphs_len)]
    graphs_train = graphs
    graphs_validate = graphs[0:int(0.2 * graphs_len)]

    # if use pre-saved graphs
    # dir_input = "/dfs/scratch0/jiaxuany0/graphs/"
    # fname_test = dir_input + args.model.name + '_' + args.dataset.name + '_' + str(args.model.num_layers) + '_' + str(
    #     args.model.hidden_size_rnn) + '_test_' + str(0) + '.dat'
    # graphs = load_graph_list(fname_test, is_real=True)
    # graphs_test = graphs[int(0.8 * graphs_len):]
    # graphs_train = graphs[0:int(0.8 * graphs_len)]
    # graphs_validate = graphs[int(0.2 * graphs_len):int(0.4 * graphs_len)]

    graph_validate_len = 0
    # for graph in graphs_validate:
    #    graph_validate_len += graph.number_of_nodes()
    # graph_validate_len /= len(graphs_validate)
    print('graph_validate_len', graph_validate_len)

    graph_test_len = 0
    # for graph in graphs_test:
    #    graph_test_len += graph.number_of_nodes()
    # graph_test_len /= len(graphs_test)
    print('graph_test_len', graph_test_len)

    args.model.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

    # args.model.max_num_node = 2000
    # show graphs statistics
    print('total graph num: {}, training set: {}'.format(len(graphs), len(graphs_train)))
    print('max number node: {}'.format(args.model.max_num_node))
    print('max/min number edge: {}; {}'.format(max_num_edge, min_num_edge))
    print('max previous node: {}'.format(args.model.max_prev_node))

    # save ground truth graphs
    ## To get train and test set, after loading you need to manually slice
    save_graph_list(graphs, args.graph_save_path + args.fname_train + '0.dat')
    save_graph_list(graphs, args.graph_save_path + args.fname_test + '0.dat')
    print('train and test graphs saved at: ', args.graph_save_path + args.fname_test + '0.dat')

    ### comment when normal training, for graph completion only
    # p = 0.5
    # for graph in graphs_train:
    #     for node in list(graph.nodes()):
    #         # print('node',node)
    #         if np.random.rand()>p:
    #             graph.remove_node(node)
    # for edge in list(graph.edges()):
    #     # print('edge',edge)
    #     if np.random.rand()>p:
    #         graph.remove_edge(edge[0],edge[1])

    ### dataset initialization
    if 'nobfs' in args.model.name:
        print('nobfs')
        dataset = Graph_sequence_sampler_pytorch_nobfs(graphs_train, max_num_node=args.model.max_num_node)
        args.model.max_prev_node = args.model.max_num_node - 1
    if 'barabasi_noise' in args.dataset.name:
        print('barabasi_noise')
        dataset = Graph_sequence_sampler_pytorch_canonical(graphs_train, max_prev_node=args.model.max_prev_node)
        args.model.max_prev_node = args.model.max_num_node - 1
    else:
        dataset = Graph_sequence_sampler_pytorch(graphs_train, max_prev_node=args.model.max_prev_node,
                                                 max_num_node=args.model.max_num_node)
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                                                                     num_samples=args.batch_size * args.batch_ratio,
                                                                     replacement=True)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                 sampler=sample_strategy)

    ### model initialization
    ## Graph RNN VAE model
    # lstm = LSTM_plain(input_size=args.model.max_prev_node, embedding_size=args.embedding_size_lstm,
    #                   hidden_size=args.hidden_size, num_layers=args.model.num_layers).to(args.device

    if 'GraphRNN_VAE_conditional' in args.model.name:
        rnn = GRU_plain(input_size=args.model.max_prev_node, embedding_size=args.model.embedding_size_rnn,
                        hidden_size=args.model.hidden_size_rnn, num_layers=args.model.num_layers, has_input=True,
                        has_output=False).to(args.device)
        output = MLP_VAE_conditional_plain(h_size=args.model.hidden_size_rnn, embedding_size=args.embedding_size_output,
                                           y_size=args.model.max_prev_node).to(args.device)
    elif 'GraphRNN_MLP' in args.model.name:
        rnn = GRU_plain(input_size=args.model.max_prev_node, embedding_size=args.model.embedding_size_rnn,
                        hidden_size=args.model.hidden_size_rnn, num_layers=args.model.num_layers, has_input=True,
                        has_output=False).to(args.device)
        output = MLP_plain(h_size=args.model.hidden_size_rnn, embedding_size=args.embedding_size_output,
                           y_size=args.model.max_prev_node).to(args.device)
    elif 'GraphRNN_RNN' in args.model.name:
        rnn = GRU_plain(input_size=args.model.max_prev_node, embedding_size=args.model.embedding_size_rnn,
                        hidden_size=args.model.hidden_size_rnn, num_layers=args.model.num_layers, has_input=True,
                        has_output=True, output_size=args.model.hidden_size_rnn_output).to(args.device)
        output = GRU_plain(input_size=1, embedding_size=args.model.embedding_size_rnn_output,
                           hidden_size=args.model.hidden_size_rnn_output, num_layers=args.model.num_layers, has_input=True,
                           has_output=True, output_size=1).to(args.device)

    ### start training
    train(args, dataset_loader, rnn, output)


def train_graphrnn(train_graphs, args):
    graphs_len = len(train_graphs)
    args.model.max_num_node = max([train_graphs[i].number_of_nodes() for i in range(graphs_len)])
    if not isinstance(args.model.max_prev_node, int):
        args.model.max_prev_node = args.model.max_num_node
    max_num_edge = max([train_graphs[i].number_of_edges() for i in range(graphs_len)])
    min_num_edge = min([train_graphs[i].number_of_edges() for i in range(graphs_len)])

    # show graphs statistics
    print('total graph num: {}'.format(graphs_len))
    print('max number node: {}'.format(args.model.max_num_node))
    print('max/min number edge: {}; {}'.format(max_num_edge, min_num_edge))
    print('max previous node: {}'.format(args.model.max_prev_node))
    dataset = Graph_sequence_sampler_pytorch(train_graphs, max_prev_node=args.model.max_prev_node,
                                             max_num_node=args.model.max_num_node)
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                                                                     num_samples=args.train.batch_size * args.train.batch_ratio,
                                                                     replacement=True)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.train.batch_size, num_workers=args.dataset.num_workers,
                                                 sampler=sample_strategy)

    if 'GraphRNN_MLP' in args.model.name:
        rnn = GRU_plain(input_size=args.model.max_prev_node, embedding_size=args.model.embedding_size_rnn,
                        hidden_size=args.model.hidden_size_rnn, num_layers=args.model.num_layers, has_input=True,
                        has_output=False).to(args.device)
        output = MLP_plain(h_size=args.model.hidden_size_rnn, embedding_size=args.model.embedding_size_output,
                           y_size=args.model.max_prev_node).to(args.device)
    elif 'GraphRNN_RNN' in args.model.name:
        rnn = GRU_plain(input_size=args.model.max_prev_node, embedding_size=args.model.embedding_size_rnn,
                        hidden_size=args.model.hidden_size_rnn, num_layers=args.model.num_layers, has_input=True,
                        has_output=True, output_size=args.model.hidden_size_rnn_output).to(args.device)
        output = GRU_plain(input_size=1, embedding_size=args.model.embedding_size_rnn_output,
                           hidden_size=args.model.hidden_size_rnn_output, num_layers=args.model.num_layers, has_input=True,
                           has_output=True, output_size=1).to(args.device)
    else:
        print("Wrong model name! please check the model name of `config/graphrnn.yaml`.")
        sys.exit(1)
    if args.train.resume:
        fname = os.path.join(args.exp_dir,
                             args.exp_name,
                             '{}_{}_{}_{}_rnn_{}.dat'.format(args.model.name,
                                                         args.dataset.name,
                                                         args.model.num_layers,
                                                         args.model.hidden_size_rnn,
                                                         args.train.resume_epoch))
        # fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
        rnn.load_state_dict(torch.load(fname))
        fname = os.path.join(args.exp_dir,
                             args.exp_name,
                             '{}_{}_{}_{}_output_{}.dat'.format(args.model.name,
                                                                args.dataset.name,
                                                                args.model.num_layers,
                                                                args.model.hidden_size_rnn,
                                                                args.train.resume_epoch))
        # fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
        output.load_state_dict(torch.load(fname))

        args.train.lr = 0.00001
        epoch = args.train.resume_epoch
        print('model loaded!, lr: {}'.format(args.train.lr))
    else:
        epoch = 1

    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.train.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.train.lr)

    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.train.milestones, gamma=args.train.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=args.train.milestones, gamma=args.train.lr_rate)

    # start main loop
    time_all = np.zeros(args.train.epochs)
    while epoch <= args.train.epochs:
        time_start = tm.time()
        # train
        if 'GraphRNN_MLP' in args.model.name:
            train_mlp_epoch(epoch, args, rnn, output, dataset,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_RNN' in args.model.name:
            train_rnn_epoch(epoch, args, rnn, output, dataset,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        if epoch % args.train.validate_epoch == 0 and epoch >= args.train.validate_epoch:
            for sample_time in range(1, 4):
                G_pred = []
                while len(G_pred) < args.train.validate_sample:
                    if 'GraphRNN_MLP' in args.model.name:
                        G_pred_step = test_mlp_epoch(epoch, args, rnn, output, test_batch_size=args.test.batch_size,
                                                     sample_time=sample_time)
                    elif 'GraphRNN_RNN' in args.model.name:
                        G_pred_step = test_rnn_epoch(epoch, args, rnn, output, test_batch_size=args.test.batch_size)
                    G_pred.extend(G_pred_step)
                # save graphs
                fname = os.path.join(args.exp_dir,
                                     args.exp_name,
                                     '{}_{}_{}_{}_pred_{}_{}.dat'.format(args.model.name,
                                                                         args.dataset.name,
                                                                         args.model.num_layers,
                                                                         args.model.hidden_size_rnn,
                                                                         epoch,
                                                                         sample_time))
                # fname = args.graph_save_path + args.fname_pred + str(epoch) + '_' + str(sample_time) + '.dat'
                save_graph_list(G_pred, fname)
                if 'GraphRNN_RNN' in args.model.name:
                    break
            print('test done, graphs saved')

        # save model checkpoint
        if args.train.save:
            if epoch % args.train.save_epoch_by == 0 and epoch >=1:
                fname = os.path.join(args.exp_dir,
                                     args.exp_name,
                                     '{}_{}_{}_{}_rnn_{}.dat'.format(args.model.name,
                                                                     args.dataset.name,
                                                                     args.model.num_layers,
                                                                     args.model.hidden_size_rnn,
                                                                     epoch))
                # fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
                torch.save(rnn.state_dict(), fname)
                fname = os.path.join(args.exp_dir,
                                     args.exp_name,
                                     '{}_{}_{}_{}_output_{}.dat'.format(args.model.name,
                                                                        args.dataset.name,
                                                                        args.model.num_layers,
                                                                        args.model.hidden_size_rnn,
                                                                        epoch))
                # fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
                torch.save(output.state_dict(), fname)
        epoch += 1
    return rnn, output


def infer_graphrnn(test_graphs=None, args=None, model=None):
    rnn, output = model
    test_batch_size = args.test.batch_size
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred = Variable(
        torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda()  # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda()  # discrete prediction
    x_step = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).cuda()
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step = output(h)
        y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=1)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).cuda()
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    # # save prediction histograms, plot histogram over each time step
    # if save_histogram:
    #     save_prediction_histogram(y_pred_data.cpu().numpy(),
    #                           fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
    #                           max_num_node=max_num_node)
    return G_pred_list
    #print("### Infer!")
import sys, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from spikingjelly.clock_driven import neuron, encoding, functional, surrogate
from os.path import join as pjoin
import sharedutils
import matplotlib.pyplot as plt

from spikingjelly.clock_driven.neuron import LIFNode, MultiStepLIFNode

from dgl import ops
from dgl.nn.functional import edge_softmax
import dgl

class Conv1dLinear(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv1dLinear, self).__init__()
        lif_params = {'tau': 1.75, 'v_threshold': 1.0, 'v_reset': 0.0}
        self.lif_neuron = LIFNode(**lif_params)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.bn(x)
        x = x.squeeze(0).permute(0, 1)
        x = self.lif_neuron(x)
        return x
    
class SpikingTransformerAttentionSepModule2(nn.Module):
    def __init__(self, dim, num_heads, n_labels, dropout, **kwargs):
        print("TransformerAttentionSepModule dim, num_heads", dim, num_heads)
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = 8

        print("TransformerAttentionSepModule dim", dim)
        self.attn_query = Conv1dLinear(dim, dim, 1)
        self.attn_key = Conv1dLinear(dim, dim, 1)
        self.attn_value = Conv1dLinear(dim, dim, 1)
        
        self.output_linear = Conv1dLinear(dim * 2, n_labels, 1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, graphA, graphx):

        queries = self.attn_query(x)
        keys = self.attn_key(x)
        values = self.attn_value(x)

        queries = queries.reshape(-1, self.num_heads, self.head_dim)
        keys = keys.reshape(-1, self.num_heads, self.head_dim)
        values = values.reshape(-1, self.num_heads, self.head_dim)

        
        #attn_scores = (queries @ keys.transpose(-2, -1))
        #attn_probs = attn_scores.softmax(dim=-1)

        #message = (attn_probs @ values) * 0.125
        #message = message.reshape(message.shape[0], -1)

        attn_scores = (queries @ keys.transpose(-2, -1))
        message = (attn_scores @ values) * 0.125
        message = message.reshape(message.shape[0], -1)


        
        x = torch.cat([x, message], axis=1)

        x = self.output_linear(x)
        x = x.permute(1, 0)

        return x
    
class SpikingTransformerAttentionSepModule(nn.Module):
    def __init__(self, dim, num_heads, n_labels, dropout, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        lif_params = {'tau': 1.75, 'v_threshold': 1.0, 'v_reset': 0.0}
        self.lif_neuron = MultiStepLIFNode(**lif_params)

        self.attn_query = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.attn_key = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.attn_value = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)

        self.output_conv = nn.Conv1d(in_channels=dim, out_channels=n_labels, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, graphA, graphx):
        #print("SpikingTransformerAttentionSepModule x0", x.shape)
        x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)  # [Batch, Channels, Time]
        #print("SpikingTransformerAttentionSepModule x1", x.shape)
        queries = self.lif_neuron(self.attn_query(x))
        keys = self.lif_neuron(self.attn_key(x))
        values = self.lif_neuron(self.attn_value(x))
        #print("SpikingTransformerAttentionSepModule queries0", queries.shape)
        #print("SpikingTransformerAttentionSepModule keys0", keys.shape)
        #print("SpikingTransformerAttentionSepModule values0", values.shape)

        queries = queries.reshape(-1, self.num_heads, self.head_dim)
        keys = keys.reshape(-1, self.num_heads, self.head_dim)
        values = values.reshape(-1, self.num_heads, self.head_dim)

        #print("SpikingTransformerAttentionSepModule queries1", queries.shape)
        #print("SpikingTransformerAttentionSepModule keys1", keys.shape)
        #print("SpikingTransformerAttentionSepModule values1", values.shape)
        
        
        #attn_scores = (queries @ keys.transpose(-2, -1))
        #attn_probs = attn_scores.softmax(dim=-1)
        
        attn_probs = (queries @ keys.transpose(-2, -1))

        x = (attn_probs @ values) * 0.125
        #print("SpikingTransformerAttentionSepModule x2", x.shape)
        #x = x.reshape(-1, self.dim * 2, x.shape[-1])
        x = x.reshape(x.shape[0], -1).unsqueeze(0)
        #x = self.dropout(self.output_conv(x))
        x = x.permute(0, 2, 1)
        #print("SpikingTransformerAttentionSepModule x3", x.shape)
        x = self.output_conv(x)
        #print("SpikingTransformerAttentionSepModule x4", x.shape)
        x = x.squeeze(0).reshape(-1, x.shape[1])
        #print("SpikingTransformerAttentionSepModule x5", x.shape)
        #x = x.permute(0, 2, 1)
        #print("SpikingTransformerAttentionSepModule x3", x.shape)
        return x

class SGTNet(nn.Module):
    def __init__(self, n_dim1, n_dim2, n_labels, tau, v_threshold):

        super(SGTNet, self).__init__()
        self.flatten = nn.Flatten()
        self.n_hidden = 64
        print("CustomNet n_dim1,n_dim2", n_dim1, n_dim2)

        self.linear0 = nn.Linear(n_dim1 * n_dim2, self.n_hidden, bias=False)
        self.transformer_attention = SpikingTransformerAttentionSepModule2(self.n_hidden, 8, n_labels, 0.5)
        #self.transformer_attention = SpikingTransformerAttentionSepModule(self.n_hidden, 8, n_labels, 0.5)
        self.lif_node = LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)

    def forward(self, x, graphA, graphx):
        x = self.flatten(x)

        x = self.linear0(x)


        x = self.transformer_attention(x, graphA, graphx)

        return x


def model_lif_fc(dataname, dataset_dir, device, batch_size,
                 learning_rate, T, tau, v_threshold, v_reset, train_epoch, log_dir, 
                 n_labels, n_dim0, n_dim1, n_dim2, train_data_loader,
                 val_data_loader, test_data_loader,graphA,graphx):
    #init

    net = SGTNet(n_dim1, n_dim2, n_labels, tau, v_threshold)

    net = net.to(device)
    # Adam opt
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.01)
    # Encoder, actually Bernoulli sampling here
    encoder = encoding.PoissonEncoder()
    train_times = 0
    max_val_accuracy = 0
    model_pth = 'tmpdir/snn/best_snn.model'
    val_accs, train_accs = [], []

    for epoch in range(train_epoch):
        net.train()
        if epoch == 50:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
        if epoch == 80:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001 
        #for rind, (img, label) in enumerate(train_data_loader):
        for rind, (img, label, batch_nodes) in enumerate(train_data_loader):
            batch_graph = dgl.node_subgraph(graphA, batch_nodes)


            img = img.to(device)

            label = label.long().to(device)
            batch_graph = batch_graph.to(device)

            label_one_hot = F.one_hot(label, n_labels).float()
            optimizer.zero_grad()

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.2)
            for t in range(T):
                if t == 0: out_spikes_counter = net(encoder(img).float(),batch_graph,graphx)
                else: out_spikes_counter += net(encoder(img).float(),batch_graph,graphx)

            out_spikes_counter_frequency = out_spikes_counter / T

            # MSE
            loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)
            loss.backward()
            optimizer.step()
            functional.reset_net(net)

            accuracy = (out_spikes_counter_frequency.max(1)[1] == label.to(device)).float().mean().item()
            
            train_accs.append(accuracy)

            train_times += 1
        scheduler.step()
        net.eval()
        with torch.no_grad():
            test_sum = 0
            correct_sum = 0

            for img, label, batch_nodes in val_data_loader:
                batch_graph = dgl.node_subgraph(graphA, batch_nodes)

                img = img.to(device)
                batch_graph = batch_graph.to(device)

                n_imgs = img.shape[0]
                out_spikes_counter = torch.zeros(n_imgs, n_labels).to(device)
                for t in range(T):
                    out_spikes_counter += net(encoder(img).float(),batch_graph,graphx)

                correct_sum += (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
                test_sum += label.numel()
                functional.reset_net(net)
            val_accuracy = correct_sum / test_sum
            val_accs.append(val_accuracy)
            if val_accuracy > max_val_accuracy:
                max_val_accuracy = val_accuracy
                torch.save(net, model_pth)
            # if  epoch ==  (train_epoch-1):
            #     torch.save(net, model_pth)
        print(f'Epoch {epoch}: device={device}, max_train_accuracy={train_accs[-1]:.4f},loss = {loss:.4f},max_val_accuracy={max_val_accuracy:.4f}, train_times={train_times}', end="\r")
    
    # 测试集：
    best_snn = torch.load(model_pth)
    best_snn.eval()
    best_snn.to(device)
    max_test_accuracy = 0.0
    result_sops, result_num_spikes_1, result_num_spikes_2 = 0, 0, 0
    with torch.no_grad():
        #functional.set_monitor(best_snn, True)
        test_sum, correct_sum = 0, 0
        for img, label, batch_nodes in test_data_loader:
            img = img.to(device)
            batch_graph = dgl.node_subgraph(graphA, batch_nodes)
            batch_graph = batch_graph.to(device)
            n_imgs = img.shape[0]
            out_spikes_counter = torch.zeros(n_imgs, n_labels).to(device)
            denominator = n_imgs * len(test_data_loader)
            for t in range(T):
                enc_img = encoder(img).float()
                out_spikes_counter += best_snn(enc_img, batch_graph, graphx)
                result_num_spikes_1 += torch.sum(enc_img) / denominator
            # post spikes
            result_num_spikes_2 += torch.sum(out_spikes_counter) / denominator
            # MSE
           
            out_spikes_counter_frequency = out_spikes_counter / T
            label = label.long().to(device)
            label_one_hot = F.one_hot(label, n_labels).float()
            loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)

            correct_sum += (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
            test_sum += label.numel()

            functional.reset_net(best_snn)

        test_accuracy = correct_sum / test_sum
        max_test_accuracy = max(max_test_accuracy, test_accuracy)
    result_msg = f'testset\'acc: device={device}, dataset={dataname}, learning_rate={learning_rate}, T={T}, max_test_accuracy={max_test_accuracy:.4f}, loss = {loss:.4f}'
    # result_msg += f", sops_per_nodes: {result_sops: .4f}"
    result_msg += f", num_s1: {int(result_num_spikes_1)}, num_s2: {int(result_num_spikes_2)}"
    result_msg += f", num_s_per_node: {int(result_num_spikes_1)+int(result_num_spikes_2)}"
    sharedutils.add_log(pjoin(log_dir, "snn_search.log"), result_msg)
    print(result_msg)
    return max_test_accuracy
import datareader, sharedutils, os
#python run_snn.py
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils import data as Data
from model_lif_fc import model_lif_fc
import dgl

def snn_run(dataname, **new_conf):
  conf, cnf = sharedutils.read_config(), {}
  cnf.update(conf['shared_conf'])
  cnf.update(conf['snn'][dataname])
  # may be some new params
  cnf.update(new_conf)
  cnf['log_dir'] = conf['snn']['log_dir']
  if cnf['v_reset'] == -100: cnf['v_reset'] = None
  print("batch_size:", cnf["batch_size"])
  rd = datareader.ReadData("~/datasets/datafromgg")

  # get fixed dataset，fixed split
  data_fixed = rd.get_fixed_splited_data(dataname)
  data =data_fixed

  print("data",data)
  mat, tag, graph = rd.conv_graph(data_fixed)
  print("mat.shape",mat.shape)
  print("tag.shape",tag.shape)
  print("graph.A.shape",graph.A.shape)
  print("graph.x.shape",graph.x.shape)

  # Convert the scipy sparse matrix to a torch sparse tensor
  A_coo = graph.A.tocoo()
  row = torch.tensor(A_coo.row).long()
  col = torch.tensor(A_coo.col).long()
  edge_index = torch.stack([row, col], dim=0)

  # Create a DGL graph from the edge_index
  dgl_graph = dgl.graph((edge_index[0], edge_index[1]))

  if dataname=="pubmed" : mat = mat+0.05
  if dataname=="citeseer" : mat = mat+0.05

  tr_ind, val_ind, ts_ind = data.split_nodes().train_nodes, \
  data.split_nodes().val_nodes, data.split_nodes().test_nodes

        
  print("train, valiadation,test's shape:", len(tr_ind), len(val_ind), len(ts_ind))
  tr_val_ind = np.hstack((tr_ind,val_ind))
     

  tr_val_mat = mat[tr_val_ind]
  tr_val_tag = tag[tr_val_ind]
  tr_mat = mat[tr_ind]
  tr_tag = tag[tr_ind]
  val_mat = mat[val_ind]
  val_tag = tag[val_ind]
  ts_mat = mat[ts_ind]
  ts_tag = tag[ts_ind]
  k = pd.DataFrame(mat)
  u = k.describe()


  self_sample = False
  
  if self_sample==True: 
    train_data_loader, val_data_loader, test_data_loader = rd.sample_numpy2dataloader(20,data_fixed, mat, tag, batch_size=cnf["batch_size"])
  else: 
    train_data_loader, val_data_loader, test_data_loader = rd.tr_ts_val_numpy2dataloader(tr_mat, ts_mat, val_mat, tr_tag,
      ts_tag, val_tag, batch_size=cnf["batch_size"])
    
       

  
  print("train, valiadation,test's batch num:", len(train_data_loader), len(val_data_loader), len(test_data_loader))
  
  n_nodes, n_feat, n_flat = mat.shape[0], mat.shape[1], 1
  print("data: %s, num_node_classes: %d" % (dataname, data.graph.num_classes))
#   print(cnf)
  ret = model_lif_fc(device=cnf["device"], dataset_dir=cnf["dataset_dir"],
                     dataname=dataname, batch_size=cnf["batch_size"], 
                     learning_rate=cnf["learning_rate"], T=cnf["T"], tau=cnf["tau"], 
                     v_reset=cnf["v_reset"], v_threshold=cnf["v_threshold"],
                     train_epoch=cnf["train_epoch"], log_dir=cnf["log_dir"], n_labels=data.graph.num_classes,
                     n_dim0=n_nodes, n_dim1=n_flat, n_dim2=n_feat, train_data_loader=train_data_loader,
                     val_data_loader=val_data_loader, test_data_loader=test_data_loader,graphA=dgl_graph,graphx=graph.x)
  
  return ret


def model_startup(dataname, runs, **new_conf):
  scores = []
  submits = []

  conc = False
  if conc:
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(10)
    for run in range(runs):
      obj = pool.submit(snn_run, dataname, **new_conf)
      submits.append(obj)
    pool.shutdown(wait=True)
    for sub in submits:
      scores.append(sub.result())
  else:
    for run in range(runs):
      score = snn_run(dataname, **new_conf)
      scores.append(score)
  sharedutils.add_log(os.path.join("./tmpdir/snn/", "snn_search.log"), "-1.5: " + str(scores))
  return np.mean(scores), np.std(scores)

def search_params(dataname, runs, log_dir):
  params_set = {"learning_rate": np.array([0.01, 0.015, 0.02, 0.025, 0.03]),
                "T": np.array([200, 300, 400, 500]),
                }

  best_score, std, best_params = sharedutils.grid_search(dataname, runs, params_set, model_startup)
  msg = "sgc; %s; best_score, std, best_params %s %s %s\n" % (dataname, best_score, std, best_params)
  print(msg)
  sharedutils.add_log(os.path.join(log_dir, "snn_search.log"), msg)

  
if __name__ == '__main__':
  print('* Set parameters in models_conf.json, such as device": "cuda:0"')
  do_search_params = input('search params? "yes" or "no", default("no"): ') or "no"
  dataname=input("cora/citeseer/pubmed, default(cora): ") or "cora"
  runs = int(input("nums of runs, default(1): ") or "1")

  if do_search_params == "yes":
    allconfs = sharedutils.read_config("./models_conf.json")
    search_params(dataname, runs, allconfs["snn"]["log_dir"])
  else:
    me, st = model_startup(dataname, runs)
    print("acc_averages %04d times: means: %04f std: %04f" % (runs, me, st))

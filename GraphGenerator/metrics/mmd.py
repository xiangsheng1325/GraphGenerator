import datetime
import numpy as np
import networkx as nx
import concurrent.futures
from functools import partial
PRINT_TIME=False


def gaussian_tv(x, y, sigma=1.0):
  support_size = max(len(x), len(y))
  # convert histogram values x and y to float, and make them equal len
  x = x.astype(np.float)
  y = y.astype(np.float)
  if len(x) < len(y):
    x = np.hstack((x, [0.0] * (support_size - len(x))))
  elif len(y) < len(x):
    y = np.hstack((y, [0.0] * (support_size - len(y))))

  dist = np.abs(x - y).sum() / 2.0# one norm
  return np.exp(-dist * dist / (2 * sigma * sigma))


def kernel_parallel_unpacked(x, samples2, kernel):
  d = 0
  for s2 in samples2:
    d += kernel(x, s2)
  return d


def kernel_parallel_worker(t):
  return kernel_parallel_unpacked(*t)


def disc(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
  ''' Discrepancy between 2 samples '''
  d = 0

  if not is_parallel:
    for s1 in samples1:
      for s2 in samples2:
        d += kernel(s1, s2, *args, **kwargs)
  else:
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #   for dist in executor.map(kernel_parallel_worker, [
    #       (s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1
    #   ]):
    #     d += dist

    with concurrent.futures.ThreadPoolExecutor() as executor:
      for dist in executor.map(kernel_parallel_worker, [
          (s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1
      ]):
        d += dist

  d /= len(samples1) * len(samples2)
  return d


def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
  ''' MMD between two samples '''
  print("--- MMD of sample1: {}, sample2:{}.---".format(len(samples1),len(samples2)))
  # normalize histograms into pmf
  if is_hist:
    samples1 = [s1 / np.sum(s1) for s1 in samples1]
    samples2 = [s2 / np.sum(s2) for s2 in samples2]
  # print('===============================')
  # print('s1: ', disc(samples1, samples1, kernel, *args, **kwargs))
  # print('--------------------------')
  # print('s2: ', disc(samples2, samples2, kernel, *args, **kwargs))
  # print('--------------------------')
  # print('cross: ', disc(samples1, samples2, kernel, *args, **kwargs))
  # print('===============================')
  return disc(samples1, samples1, kernel, *args, **kwargs) + \
          disc(samples2, samples2, kernel, *args, **kwargs) - \
          2 * disc(samples1, samples2, kernel, *args, **kwargs)


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)
    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv, sigma=2.0)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


def print_result(metrics, graph_ref, graph_pred):
    output = {}
    if 'degree' in metrics:
        eval_metric = degree_stats(graph_ref, graph_pred)
        print('Degree: {}'.format(eval_metric))
        output['degree']=eval_metric
    return output


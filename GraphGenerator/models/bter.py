import os
import numpy as np
import networkx as nx
from GraphGenerator.utils.data_utils import load_matlab_graph, save_matlab_graph


def generate_matlab_mat(data_name, in_mat_path, out_mat_path, repeat=2):
    bter_path = "./GraphGenerator/models/bter_ops/"
    template_filename = os.path.join(bter_path, "template.m")
    with open(template_filename, "r") as r_f:
        template_context = r_f.read()
    tmp_filepath = os.path.join(bter_path, "{}.m".format(data_name))
    with open(tmp_filepath, "w") as w_f:
        context = template_context.split("%##{Template Block}##%")
        w_f.write(context[0])
        w_f.write(os.path.join("../../../", in_mat_path))
        w_f.write(context[1])
        w_f.write(str(repeat))
        w_f.write(context[2])
        w_f.write(os.path.join("../../../", out_mat_path))
        w_f.write(context[3])
    os.system("matlab -nosplash -nodesktop -r "
              "'cd ./GraphGenerator/models/bter_ops; {}; cd ../../..; quit'".format(data_name))
    graphs = load_matlab_graph(fname=out_mat_path)
    # print(graphs)
    os.remove(tmp_filepath)
    return graphs


def bter(input_graph, config):
    fname = os.path.join(config.exp_dir, config.exp_name, "{}.mat".format(config.dataset.name))
    dump_name = os.path.join(config.exp_dir, config.exp_name, "bter_to_{}.mat".format(config.dataset.name))
    sp_adj = nx.adjacency_matrix(input_graph)
    sp_adj.data = sp_adj.data.astype(np.float64)
    save_matlab_graph(fname, sp_adj, config.dataset.name)
    graphs = generate_matlab_mat(data_name=config.dataset.name,
                                 in_mat_path=fname,
                                 out_mat_path=dump_name,
                                 repeat=config.num_gen)
    # print(graphs)
    return [nx.Graph(graph) for graph in graphs[0].tolist()]


if __name__ == '__main__':
    tmp_g = nx.grid_2d_graph(10, 10)
    save_matlab_graph("./tmp.mat", nx.adjacency_matrix(tmp_g), "tmp")

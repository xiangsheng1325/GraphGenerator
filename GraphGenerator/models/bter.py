import os
from GraphGenerator.utils.data_utils import load_matlab_graph, save_matlab_graph


def generate_matlab_script(data_name, in_mat_path, out_mat_path):
    bter_path = "./GraphGenerator/models/bter_ops/"
    template_filename = os.path.join(bter_path, "template.m")
    with open(template_filename, "r") as r_f:
        template_context = r_f.read()
    tmp_filepath = os.path.join(bter_path, "{}.m".format(data_name))
    with open(tmp_filepath, "w") as w_f:
        context = template_context.split("###{Template Block}###")
        w_f.write(context[0])
        w_f.write(os.path.join("../../../", in_mat_path))
        w_f.write(context[1])
        w_f.write(os.path.join("../../../", out_mat_path))
        w_f.write(context[2])


def bter(input_graph, config):
    fname = os.path.join(config.exp_dir, config.exp_name, "{}.mat".format(config.data.name))
    dump_name = os.path.join(config.exp_dir, config.exp_name, "bter_to_{}.mat".format(config.data.name))
    save_matlab_graph(fname, input_graph, config.data.name)
    generate_matlab_script(data_name=config.data.name,
                           in_mat_path=fname,
                           out_mat_path=dump_name)
    os.system("matlab 'cd ./GraphGenerator/models/bter_ops; {}; cd ../../..; quit'".format(config.data.name))
    G, _ = load_matlab_graph(fname=dump_name)
    return G

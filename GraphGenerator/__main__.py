import argparse, sys, pickle, warnings, os, torch

# import torch.cuda

warnings.filterwarnings("ignore")
from GraphGenerator.preprocessing import dataio
from GraphGenerator.utils.arg_utils import get_config, set_device
import pandas as pd


def print_variables(vdict, name="args"):
    print("-----------------------------------------")
    print("|This is the summary of {}:".format(name))
    var = vdict
    for i in var:
        if var[i] is None:
            continue
        print("|{:11}\t: {}".format(i, var[i]))
    print("-----------------------------------------")


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--phase", help="Choose phase.", default="preprocessing", type=str,
                        choices=["preprocessing", "train", "evaluate", "test"],
                        required=True)
    parser.add_argument("-i", "--input", help="Path of input file. Example:```-i google.txt```", default=None)
    parser.add_argument("-o", "--output", help="Specify the name of output file.", default=None)
    parser.add_argument("-c", "--config", help="Specify the path of config file.", default=None)
    parser.add_argument("-g", "--generator", help="choose the generator. Example:```-g sbm```", default="vgae",
                        choices=["e-r", "b-a", "w-s", "rtg", "bter", "sbm", "dcsbm", "rmat", "kronecker",
                                 "mmsb", "vgae", "graphite", "sbmgnn", "graphrnn", "gran", "bigg", "arvga",
                                 "netgan", "condgen", "sgae"])
    parser.add_argument("-e", "--evaluate", help="choose the evaluating metrics.", default=None)
    parser.add_argument("-r", "--ref", help="Path of referenced graphs(Only required in evaluate phase)", default=None)
    args = parser.parse_args()
    print_variables(vars(args))
    if args.phase == 'preprocessing':
        from GraphGenerator.preprocessing import utils
        tmp_path = args.input
        print("# Load edgelist...")
        graph = utils.edgelist_to_graph(tmp_path)
        graphlist = [graph]
        print("# Save graphlist...")
        if args.output is None:
            output_name = "{}.graphs".format(args.input)
        else:
            output_name = args.output
        dataio.save_data(graphlist, name=output_name)

    elif args.phase == 'train':
        config = get_config(args.config)
        set_device(config)
        from GraphGenerator.train import train_base as train
        print("Start loading data...")
        input_data = dataio.load_data(args.input)
        if args.config is None:
            args.config = "config/{}.yaml".format(args.generator)
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
        print("Start (training and) inferencing graph...")
        output_data = []
        if isinstance(input_data, list):
            for graph in input_data:
                tmp_data = train.train_and_inference(graph, args.generator, config=config)
                if isinstance(tmp_data, list):
                    output_data.extend(tmp_data)
                else:
                    output_data.append(tmp_data)
        else:
            tmp_data = train.train_and_inference(input_data, args.generator)
            if isinstance(tmp_data, list):
                output_data.extend(tmp_data)
            else:
                output_data.append(tmp_data)
        print("Start saving generated graphs...")
        if args.output is None:
            output_name = "{}_to_{}.graphs".format(config.dataset.name, args.generator)
        else:
            output_name = args.output
        dataio.save_data(output_data, name=os.path.join(config.exp_dir, config.exp_name, output_name))
    elif args.phase == 'evaluate':
        config = get_config(args.config)
        set_device(config)
        if args.evaluate == 'efficiency':
            from GraphGenerator.evaluate.efficiency import eval_efficiency
            print("Start evaluating the efficiency of graph generator [{}].".format(args.generator))
            result = eval_efficiency(args.generator, config)
        elif args.evaluate == 'performance':
            from GraphGenerator.metrics import mmd
            print("Start evaluating the quality of generated graphs...")
            graphs_ref = dataio.load_data(args.ref)
            graphs_pred = dataio.load_data(args.input)
            result = mmd.print_result(args.evaluate, graphs_ref, graphs_pred)
            if args.output is None:
                output_name = "{}_to_{}.csv".format(args.ref, args.input)
            else:
                output_name = args.output
            tmp_pd = pd.DataFrame(result)
            tmp_pd.to_csv(output_name)
    elif args.phase == 'test':
        config = get_config(args.config)
        set_device(config)
        from GraphGenerator.test import test_generator
        print("Start test the package...")
        test_generator(args, config)
        print("Memory reserved: {} KiB.".format(torch.cuda.memory_reserved(config.device)//1024))
        print("Test finished.")
    print("Done!")
    # sys.exit(0)

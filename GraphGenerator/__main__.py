import GraphGenerator.metrics.mmd
import GraphGenerator.train
import GraphGenerator.preprocessing.load_data
import argparse


def print_variables(vdict, name="args"):
    print("┌---------------------------------------")
    print("|This is the summary of {}:".format(name))
    var = vdict
    for i in var:
        print("|{:11}\t: {}".format(i, var[i]))
    print("└---------------------------------------")


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--generator", help="choose the generator. Example:sbm, dcsbm", default='sbm')
    parser.add_argument("-d", "--dataset", help="choose the dataset 'cora'/'citeseer'/'pubmed'", default='cora')
    parser.add_argument("-s", "--save", help="Bool(whether to save generated graphs.)", default=True, type=bool)
    parser.add_argument("-e", "--evaluate", help="choose the evaluating metrics.", default='degree')
    args = parser.parse_args()
    print_variables(vars(args))
    print("Start loading data...")
    input_data = GraphGenerator.preprocessing.load_data.load_data(args.dataset)
    print("Start (training and) inferencing graph...")
    output_data = GraphGenerator.train.train_and_inference(input_data, args.generator)
    print("Start evaluating the generated graphs...")
    GraphGenerator.metrics.mmd.print_result(args.evaluate, [input_data], output_data)
    print("Start saving generated graphs...")
    GraphGenerator.preprocessing.load_data.save_data(output_data, name=args.generator)
    print("Done!")

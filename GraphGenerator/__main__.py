import argparse, sys
from GraphGenerator.metrics import mmd
from GraphGenerator import train
from GraphGenerator.preprocessing import load_data


def print_variables(vdict, name="args"):
    print("┌---------------------------------------")
    print("|This is the summary of {}:".format(name))
    var = vdict
    for i in var:
        if i is None:
            continue
        print("|{:11}\t: {}".format(i, var[i]))
    print("└---------------------------------------")


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--phase", help="Choose phase.", default="preprocessing", type=str,
                        choices=["preprocessing"],
                        required=True)
    parser.add_argument("-g", "--generator", help="choose the generator. Example:sbm, dcsbm", default=None)
    parser.add_argument("-d", "--dataset", help="choose the dataset 'cora'/'citeseer'/'pubmed'", default=None)
    parser.add_argument("-s", "--save", help="Bool(whether to save generated graphs.)", default=None)
    parser.add_argument("-e", "--evaluate", help="choose the evaluating metrics.", default=None)
    args = parser.parse_args()
    print_variables(vars(args))
    if args.phase == 'preprocessing':
        print("Preprocessing, yes!")

        sys.exit(0)
    print("Start loading data...")
    input_data = load_data.load_data(args.dataset)
    print("Start (training and) inferencing graph...")
    output_data = train.train_and_inference(input_data, args.generator)
    print("Start evaluating the generated graphs...")
    mmd.print_result(args.evaluate, [input_data], output_data)
    print("Start saving generated graphs...")
    load_data.save_data(output_data, name=args.generator)
    print("Done!")

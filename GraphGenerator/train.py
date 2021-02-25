import GraphGenerator.models.sbm
import sys


def train_and_inference(input_data, generator, repeat=1):
    graphs = []
    if generator in ['sbm', 'dcsbm']:
        graphs = GraphGenerator.models.sbm.generate(input_data, generator, repeat)
    else:
        print("Wrong generator name! Process exit..")
        sys.exit(1)
    return graphs


import GraphGenerator.models.sbm


def train_and_inference(input_data, generator, repeat=1):
    graphs = []
    if generator in ['sbm', 'dcsbm']:
        graphs = GraphGenerator.models.sbm.generate(input_data, generator, repeat)
    return graphs


import pickle, os, sys


def load_data(path):
    if os.path.exists(path):
        graph = pickle.load(open(path, "rb"))
        return graph
    else:
        print("Invalid input data...")
        sys.exit(1)


def save_data(obj, name):
    pickle.dump(obj, open("{}".format(name), "wb"))
    return 0


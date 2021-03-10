from .test_bigg import bigg_test as bigg


def test_generator(args):
    eval(args.generator)(args)
    return

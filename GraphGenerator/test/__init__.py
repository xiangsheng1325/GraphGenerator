from .test_bigg import bigg_test as bigg


def test_generator(args, config):
    eval(args.generator)(args, config)
    return

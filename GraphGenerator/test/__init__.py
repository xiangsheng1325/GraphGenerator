from .test_bigg import bigg_test


def test_package(args):
    if args.generator == 'bigg':
        bigg_test(args)
    return

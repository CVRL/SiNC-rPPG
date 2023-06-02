import sys

def get_dataset(split, arg_obj):
    dataset = arg_obj.dataset.lower()

    if dataset == 'pure_unsupervised':
        from datasets.PURE_unsupervised import PUREUnsupervised as DataSet
        print('Using PURE unsupervised dataset.')
    elif dataset == 'pure_supervised':
        from datasets.PURE_supervised import PURESupervised as DataSet
        print('Using PURE supervised dataset.')
    elif dataset == 'pure_testing':
        from datasets.PURE_testing import PURESupervised as DataSet
        print('Using PURE testing dataset.')

    else:
        print('Dataset not found. Exiting.')
        sys.exit(-1)

    return DataSet(split, arg_obj)

import sys

def get_dataset(split, arg_obj):
    dataset = arg_obj.dataset.lower()

    if dataset == 'ddpm_unsupervised':
        from datasets.DDPM_unsupervised import DDPMUnsupervised as DataSet
        print('Using DDPM unsupervised dataset.')
    elif dataset == 'ddpm_supervised':
        from datasets.DDPM_supervised import DDPMSupervised as DataSet
        print('Using DDPM supervised dataset.')
    elif dataset == 'ddpm_testing':
        from datasets.DDPM_testing import DDPMSupervised as DataSet
        print('Using DDPM testing dataset.')

    elif dataset == 'ubfc_unsupervised':
        from datasets.UBFC_unsupervised import UBFCUnsupervised as DataSet
        print('Using UBFC unsupervised dataset.')
    elif dataset == 'ubfc_supervised':
        from datasets.UBFC_supervised import UBFCSupervised as DataSet
        print('Using UBFC supervised dataset.')
    elif dataset == 'ubfc_testing':
        from datasets.UBFC_testing import UBFCSupervised as DataSet
        print('Using UBFC testing dataset.')

    elif dataset == 'pure_unsupervised':
        from datasets.PURE_unsupervised import PUREUnsupervised as DataSet
        print('Using PURE unsupervised dataset.')
    elif dataset == 'pure_supervised':
        from datasets.PURE_supervised import PURESupervised as DataSet
        print('Using PURE supervised dataset.')
    elif dataset == 'pure_testing':
        from datasets.PURE_testing import PURESupervised as DataSet
        print('Using PURE testing dataset.')

    elif dataset == 'celebv_unsupervised':
        from datasets.CelebV_unsupervised import CelebV as DataSet
        print('Using CelebV unsupervised dataset.')

    elif dataset == 'hkbu_unsupervised':
        from datasets.HKBU_unsupervised import HKBU as DataSet
        print('Using HKBU unsupervised dataset.')

    else:
        print('Dataset not found. Exiting.')
        sys.exit(-1)

    return DataSet(split, arg_obj)

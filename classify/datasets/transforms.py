import albumentations
from torchvision.transforms import transforms

al = albumentations
tr = transforms


def build_transform(seq, args, kwargs):
    """
    demo:
    seq = ['al.RandomSnow', 'albumentations.Blur', 'transforms.CenterCrop']
    arg = [[],[],[]]
    dic = [{}, {}, {'size':224}]
    seq = [eval(s)(*a, **d) for s, a, d in zip(seq, args, kwargs)]
    seq = tr.Compose(seq)
    return seq
    :param seq:
    :param args:
    :param kwargs:
    :return:
    """
    seq = [eval(s)(*a, **d) for s, a, d in zip(seq, args, kwargs)]
    seq = tr.Compose(seq)
    return seq


if __name__ == '__main__':
    from pprint import pprint

    pprint(albumentations.__dict__)
    pprint(transforms.__dict__)

    seq = ['al.RandomSnow', 'al.Blur', 'transforms.CenterCrop']
    args = [[], [], []]
    kwargs = [{}, {}, {'size': 224}]
    seq = build_transform(seq, args, kwargs)
    print(seq)

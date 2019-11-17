
train_datasets = [
    dict(type='ImageFolder',
         root='/home/cmf/datasets/lfw/lfw-a')
]

val_datasets = [
    dict(type='ImageFolder',
         root='/home/cmf/datasets/lfw/lfw-a')
]

# import torchvision
#
# transform = [
#     # torchvision.transforms.FiveCrop(50)
#     torchvision.transforms.ColorJitter()
#     dict(type='FiveCrop', size=50),
#     dict(type='ToTensor')
#
# ]
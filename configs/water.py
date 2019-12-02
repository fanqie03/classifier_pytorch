train_datasets = [
    dict(type='ImageFolder',
         root='/home/cmf/datasets/water/train_val/train'),
    dict(type='ImageFolder',
         root='/home/cmf/datasets/water/train_val2/train'),
    # dict(type='ImageFolder',
    #      root='/home/cmf/w_public/mmdetection/data/extract_video_tayg_0.97/train'),
    # dict(type='ImageFolder',
    #      root='/home/cmf/w_public/mmdetection/data/extract_video_tayg_0.97/train'),
    # dict(type='ImageFolder',
    #      root='/home/cmf/datasets/helmet/helmet-wash/train')
]

val_datasets = [
    dict(type='ImageFolder',
         root='/home/cmf/datasets/water/train_val/val'),
    dict(type='ImageFolder',
         root='/home/cmf/datasets/water/train_val2/val'),
]

scheduler = dict(
    type='torch.optim.lr_scheduler.StepLR',
    step_size=10,
    gamma=0.1,
)

model = dict(
    type='mobilenet_v2',
    num_classes=2
)

train = dict(
    pretrained_model=None,
    freeze_body=False,
    freeze_head=False,
    resume_from=None,
    init_type='xavier',
    work_dir='checkpoint',
    num_epochs=200,
    batch_size=64,
    start_epoch=0,
    use_tensorboard=False,
    val_epochs=1,
    open_tensorboard=True
)

optimizer = dict(
    type='torch.optim.SGD',
    lr=0.001,
    momentum=0.9
)

loss = dict(
    type='nn.CrossEntropyLoss',
    # weight=[1, 1.5]
)

#  注释性质
data = dict(
    input_size=(224, 224),
    type='rgb',
    mean=0,
    std=255
)

transform = [
    # torchvision.transforms.FiveCrop(50)
    # torchvision.transforms.ColorJitter()
    # dict(type='RandomAffine', degrees=30, scale=(0.5, 1.5), shear=5),
    dict(type='RandomHorizontalFlip'),
    # dict(type='ColorJitter', brightness=0.3, contrast=0.3, saturation=0.3),
    # dict(type='ColorJitter', brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    # dict(type='RandomGrayscale'),
    # dict(type='iaa.CoarseDropout', p=(0.03, 0.15), size_percent=(0.1, 0.4)),
    dict(type='Resize', size=(224, 224)),
    dict(type='ToTensor'),
    # dict(type='RandomErasing', ratio=(10, 15), inplace=True),
    # dict(type='RandomErasing', ratio=(10, 15), inplace=True),
    # dict(type='RandomErasing', ratio=(0.1, 0.2), inplace=True),
    # dict(type='RandomErasing', ratio=(0.1, 0.2), inplace=True),

]

val_transform = [
    dict(type='Resize', size=(224, 224)),
    dict(type='ToTensor'),
]

dataloader = dict(
    batch_size=64,
    num_workers=6,
    shuffle=True
)

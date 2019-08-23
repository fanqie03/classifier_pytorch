
model = dict(
    type='resnet18'
)

train_cfg = dict(

)

data = dict(
    batch_size=32,
    num_workers=4,
    train=dict(
        img_scale=(224, 224)
    ),
    val=dict(
        img_scale=(224, 224)
    ),
    test=dict(
        img_scale=(224, 224)
    )
)

img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

total_epochs = 12
work_dir = 'checkpoint'
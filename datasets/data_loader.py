from .monuseg import MoNuSeg
from .liver_paip2019 import LiverC
from torch.utils import data

def get_loader(image_path, anno_path, batch_size, name_dataset = 'paip', num_workers=3, data_aug=True, prob=0.5, transform=None):
    """Builds and returns Dataloader."""
    if name_dataset == 'paip':
        dataset = LiverC(img_root = image_path, anno_root = anno_path, data_aug=data_aug, prob=prob, transform=transform)
    elif name_dataset == 'monuseg':
        dataset = MoNuSeg(img_root = image_path, anno_root = anno_path, data_aug=data_aug, prob=prob, transform=transform)
    else:
        raise NotImplementedError(name_dataset+" is not implemented, support {'paip, monuseg'} now.")
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers, drop_last=True)
    return data_loader

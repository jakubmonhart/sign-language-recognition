from data import WLASL

from torchvision import transforms
import videotransforms

from torch.utils.data import DataLoader

# Train dataset transforms
train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip()])

# Create dataset
train_dataset = WLASL(
    json_file='data/WLASL_v0.3.json', videos_path='data/sample-videos',
    split='train', transforms=train_transforms)

train_dataloader = DataLoader(
    train_dataset, batch_size=1, shuffle=True, num_workers=0)

# Try to load videos
for b in train_dataloader:
    print(b['X'].shape)


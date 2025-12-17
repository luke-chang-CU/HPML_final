from data_utils import MiniImageNetDataset

print("Building palette for zh-plus/tiny-imagenet...")
dataset = MiniImageNetDataset(split='train', build_palette=True)
print("Palette built.")

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.datasets as ds
from torchvision import transforms


class ImageNetProcessor:
    def __init__(
        self,
        num_samples=1000,
        calibration_samples=100,
        batch_size=32,
        num_workers=4,
        image_size=224,
        seed=42,
        cache_dir=None,
    ):
        self.num_samples = num_samples
        self.calibration_samples = calibration_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # Set the global seed for Torch
        torch.manual_seed(self.seed)

        # Load the validation set in streaming mode
        dataset = load_dataset(
            "timm/imagenet-1k-wds", split="validation", streaming=True, cache_dir=cache_dir
        )

        # Shuffle the dataset and take required samples
        shuffled_dataset = dataset.shuffle(seed=seed)
        self.main_dataset = shuffled_dataset.take(num_samples)
        self.calibration_dataset = shuffled_dataset.skip(num_samples).take(calibration_samples)

        # Define the transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Create the main dataloader
        self.dataloader = self._create_dataloader(self.main_dataset)

        # Create the calibration dataloader
        self.calibration_dataloader = self._create_dataloader(self.calibration_dataset)

    def preprocess(self, example):
        example["pixel_values"] = self.transform(example["jpg"].convert("RGB"))
        return example

    def get_calibration_tensor(self):
        # Process all calibration samples
        calibration_samples = [self.preprocess(example) for example in self.calibration_dataset]

        # Stack all preprocessed images into a single tensor
        calibration_tensor = torch.stack([sample["pixel_values"] for sample in calibration_samples])

        return calibration_tensor

    def _create_dataloader(self, dataset):
        def collate_fn(examples):
            processed_examples = [self.preprocess(example) for example in examples]
            pixel_values = torch.stack([example["pixel_values"] for example in processed_examples])
            labels = torch.tensor([example["cls"] for example in processed_examples])
            return {"pixel_values": pixel_values, "labels": labels}

        # Create a Generator with a fixed seed
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        return DataLoader(
            list(dataset),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            shuffle=True,
            generator=generator,
            worker_init_fn=lambda worker_id: torch.manual_seed(self.seed + worker_id),
        )

    @staticmethod
    def compute_accuracy(outputs, targets, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = targets.size(0)

            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    @classmethod
    def accuracy(cls, outputs, targets):
        return cls.compute_accuracy(outputs, targets, topk=(1,))[0].item()

    @classmethod
    def accuracy_top5(cls, outputs, targets):
        return cls.compute_accuracy(outputs, targets, topk=(5,))[0].item()
class CIFAR10Processor:
    def __init__(
        self,
        num_samples=1000,
        calibration_samples=100,
        batch_size=32,
        num_workers=4,
        seed=42,
        data_dir='./data'
    ):
        self.num_samples = num_samples
        self.calibration_samples = calibration_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        torch.manual_seed(self.seed)

        # CIFAR-10 transforms - NO RESIZING, keep at 32x32
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 specific
                std=[0.2023, 0.1994, 0.2010]
            ),
        ])

        # Load CIFAR-10 test set

        full_dataset = ds(
            root=data_dir, 
            train=False, 
            download=True, 
            transform=self.transform
        )

        indices = torch.randperm(len(full_dataset), generator=torch.Generator().manual_seed(seed)).tolist()
        main_indices = indices[:num_samples]
        calib_indices = indices[num_samples:num_samples + calibration_samples]

        self.main_dataset = torch.utils.data.Subset(full_dataset, main_indices)
        self.calibration_dataset = torch.utils.data.Subset(full_dataset, calib_indices)

        generator = torch.Generator().manual_seed(seed)
        self.dataloader = DataLoader(
            self.main_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            generator=generator,
            collate_fn=self._collate_fn
        )

        self.calibration_dataloader = DataLoader(
            self.calibration_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        return {"pixel_values": images, "labels": labels}

    def get_calibration_tensor(self):
        all_images = []
        for batch in self.calibration_dataloader:
            all_images.append(batch["pixel_values"])
        return torch.cat(all_images)

    @staticmethod
    def compute_accuracy(outputs, targets, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = targets.size(0)
            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    @classmethod
    def accuracy(cls, outputs, targets):
        return cls.compute_accuracy(outputs, targets, topk=(1,))[0].item()

    @classmethod
    def accuracy_top5(cls, outputs, targets):
        return cls.compute_accuracy(outputs, targets, topk=(5,))[0].item()
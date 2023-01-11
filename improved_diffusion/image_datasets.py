from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    import ipdb; ipdb.set_trace()
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir) # 找到所有的图片, e.g., len(all_files)=50000
    classes = None # 图片的类别
    if class_cond: # True
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files] # e.g., 'bird', 这是获取一个图片的分类标签，例如bird, ...
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))} # {'bird': 0, 'car': 1, 'cat': 2, 'deer': 3, 'dog': 4, 'frog': 5, 'horse': 6, 'plane': 7, 'ship': 8, 'truck': 9}
        classes = [sorted_classes[x] for x in class_names] # 50000个元素，从0到9
    dataset = ImageDataset(
        image_size, #64
        all_files, # 50000, e.g., '/workspace/asr/diffusion_models/improved-diffusion/datasets/cifar_train/bird_00006.png'
        classes=classes, # 50000, from 0 to 9
        shard=MPI.COMM_WORLD.Get_rank(), # 0
        num_shards=MPI.COMM_WORLD.Get_size(), # 1, 只有一张卡
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True # 1 -> 0
        )
    else: # HERE, deterministic=False, so, shuffle=True
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True # 1 -> 0
        )
    import ipdb; ipdb.set_trace()
    while True:
        yield from loader
    # TODO why can not in 'load_data' in ipdb mode? strange
    # 有意思，去掉上面的两行，yield就可以了，先debug了再说！！！

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results # 找到了50000张图片, e.g., '/workspace/asr/diffusion_models/improved-diffusion/datasets/cifar_train/bird_00006.png'
    # 遍历到所有的图片类型。 TODO

class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        #import ipdb; ipdb.set_trace() # ImageDataset init
        super().__init__()
        self.resolution = resolution # 64
        self.local_images = image_paths[shard:][::num_shards] # 5000
        self.local_classes = None if classes is None else classes[shard:][::num_shards] # 50000 个元素，从0到9.

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        import ipdb; ipdb.set_trace()
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality. 手动缩小图片
        while min(*pil_image.size) >= 2 * self.resolution: # (32, 32) compares with 64=self.resolution
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size) # e.g., 64/32 = 2.0
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        ) # 这是把原来的图片，原地放大到原来的二倍.
        # 图片做一些归一化, TODO
        arr = np.array(pil_image.convert("RGB")) # (64, 64, 3)
        crop_y = (arr.shape[0] - self.resolution) // 2 # 0
        crop_x = (arr.shape[1] - self.resolution) // 2 # 0
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution] # [64, 64, 3]
        arr = arr.astype(np.float32) / 127.5 - 1 # [64, 64, 3]

        out_dict = {}
        if self.local_classes is not None: # len(self.local_classes) = 50000， 五万张图片
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict # from [64, 64, 3] to [3, 64, 64]; out_dict = {'y': array(1)}

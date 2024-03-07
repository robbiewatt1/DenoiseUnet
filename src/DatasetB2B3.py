import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset
import h5py


class EmittanceBlur(object):
    """
    Applies a random blur to the image with to account for emittance effects
    """

    def __init__(self, max_blur):
        """
        :param max_blur: maximum blur size
        """
        self.max_blur = max_blur

    def __call__(self, image):
        """
        Forward call applying transform
        :param image: image to apply transform to
        """
        sigma = np.random.randint(1, self.max_blur // 2, 2) * 2 + 1
        return torchvision.transforms.functional.gaussian_blur(
            image, list(sigma))


class RandomCrop(object):
    """
    Randomly crops the image to a similar size as the real data
    """

    def __init__(self, crop_size, centre_shit=(10, 100)):
        """
        :param crop_size: size of the crop
        :param image_size: size of the image
        """
        self.crop_size = crop_size
        self.centre_shift = centre_shit

    def __call__(self, image):
        x_centre = np.random.randint(-self.centre_shift[0]//2,
                                     self.centre_shift[0]//2)
        y_centre = np.random.randint(-self.centre_shift[1]//2,
                                     self.centre_shift[1]//2)
        x_start = (image.shape[-2] - self.crop_size[0]) // 2 + x_centre
        y_start = (image.shape[-1] - self.crop_size[1]) // 2 + y_centre

        return image[..., x_start:x_start+self.crop_size[0],
                     y_start:y_start+self.crop_size[1]]


class BackgroundTrans(object):
    """
    Applies a background noise to the image
    """

    def __init__(self, mean_range=(6., 9.), sig_range=(6., 9.)):
        """
        :param mean_range: mean of the background noise
        :param sig_range: standard deviation of the background noise
        """
        self.mean_range = mean_range
        self.sig_range = sig_range

    def __call__(self, image):
        """
        Forward call applying transform
        :param image: image to apply transform to
        """
        # Sample background noise
        mean = np.random.uniform(*self.mean_range)
        sig = np.random.uniform(*self.sig_range)
        noise = torch.normal(torch.ones_like(image) * mean,
                             torch.ones_like(image) * sig)
        noise = torch.where(noise < 0, 0, noise)
        return image + noise


class PoissonTrans(object):
    """
    Max normalisation and apply a poisson distribution
    """

    def __init__(self, count_range=(3, 20)):
        """
        :param count_range: range of max photon counts to sample from
        """
        self.count_range = count_range

    def __call__(self, image):
        """
        Forward call applying transform
        :param image: image to apply transform to
        """
        # Sample photon counts
        image = image * np.random.randint(self.count_range[0],
                                          self.count_range[1])
        return torch.poisson(image)


class DenoiseDataset(Dataset):
    """
    Dataset for denoising images for BC11 from simulations
    """

    def __init__(self, data_path, input_norm=1.1e8, blur_max=101,
                 affine_params=(10., (0.05, 0.05)),
                 count_lims=(5, 20), background_params=((7., 9.), (3., 4.)),
                 norm_params=(11., 7.),
                 device='cpu'):
        """
        :param input_norm: normalisation of the input data
        :param data_path: path to the data
        :param blur_max: maximum blur size
        :param affine_params: affine parameters for random affine transform
        :param count_lims: range of max photon counts to sample from
        :param background_params: mean and standard deviation of the background
        :param resize: size to resize the image to
        :param norm_params: mean and standard deviation of the normalisation
        :param device: device to load the data on
        """

        file = h5py.File(data_path, 'r')
        self.images = file["images"]
        self.norm_params = norm_params
        self.device = device

        self.base_trans = torchvision.transforms.Compose(
            [torchvision.transforms.Lambda(lambda t: t / input_norm),
             EmittanceBlur(blur_max),
             torchvision.transforms.RandomAffine(
                 affine_params[0], affine_params[1],
             interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
             torchvision.transforms.Lambda(
                 lambda t: t[..., 302:-402, 327:-327]),
             torchvision.transforms.RandomVerticalFlip(),
             torchvision.transforms.RandomHorizontalFlip()])

        self.input_trans = torchvision.transforms.Compose(
            [PoissonTrans(count_lims),
             BackgroundTrans(*background_params),
             torchvision.transforms.Lambda(
                 lambda t: (t - norm_params[0]) / norm_params[1])])

    def __len__(self):
        """
        :return: length of the dataset
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        :param index: index of the dataset
        :return: input data and time data
        """
        image = self.images[index]
        image = torch.tensor(image).float().to(self.device)[None]

        base_image = self.base_trans(image)
        return (self.input_trans(base_image),
                base_image)

    def reverse_transform(self, image):
        """
        Reverse the transforms applied to the image.
        :param image: image to reverse transforms on
        """
        return (image * self.norm_params[1]) + self.norm_params[0]


class DatasetTest(Dataset):
    """
    Dataset for testing the denoising network
    """

    def __init__(self, data_path, norm_params=(11., 7.),
                 device='cpu'):
        """
        :param data_path: path to the data
        :param norm_params: mean and standard deviation of the normalisation
        :param device: device to load the data on
        """
        file = h5py.File(data_path, 'r')
        self.images = file["Images"]
        self.norm_params = norm_params
        self.device = device

    def __len__(self):
        """
        :return: length of the dataset
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        :param index: index of the dataset
        :return: input data and time data
        """
        images = self.images[index]
        images = torch.tensor(np.array(images)).float().to(self.device)
        images_mean = torch.mean(images, dim=2)

        return ((images[..., 300:556, 300:556] - self.norm_params[0])
                / self.norm_params[1], images_mean[..., 300:556, 300:556])

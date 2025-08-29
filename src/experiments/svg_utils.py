import math
import torch
import socket
import argparse
import os
import numpy as np
from sklearn.manifold import TSNE
try:
    import scipy.misc
    # Test if toimage is available
    if not hasattr(scipy.misc, 'toimage'):
        raise ImportError("scipy.misc.toimage not available")
except ImportError:
    # scipy.misc is deprecated, create replacement
    import numpy as np
    from PIL import Image
    
    class SciPyMiscReplacement:
        @staticmethod
        def toimage(arr, high=255, channel_axis=0):
            """Replacement for deprecated scipy.misc.toimage"""
            arr = np.array(arr)
            
            # Handle channel axis
            if channel_axis == 0 and arr.ndim == 3:
                arr = arr.transpose(1, 2, 0)
            
            # Normalize to 0-255 range
            if arr.max() > 0:
                arr = (arr * high / arr.max()).astype('uint8')
            else:
                arr = arr.astype('uint8')
            
            # Handle different array shapes
            if arr.ndim == 3:
                if arr.shape[2] == 1:
                    arr = arr.squeeze(axis=2)  # Remove single channel dimension
                elif arr.shape[2] == 3:
                    pass  # RGB image
                else:
                    # Convert single channel to RGB
                    arr = np.stack([arr] * 3, axis=2)
            elif arr.ndim == 2:
                pass  # Grayscale image
            
            return Image.fromarray(arr)
    
    # Replace scipy.misc with our implementation
    import scipy
    if not hasattr(scipy, 'misc'):
        scipy.misc = SciPyMiscReplacement()
    else:
        scipy.misc.toimage = SciPyMiscReplacement.toimage
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import functools
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_metric
    from skimage.metrics import structural_similarity as ssim_metric
except ImportError:
    # Fallback for older skimage versions
    from skimage.measure import compare_psnr as psnr_metric
    from skimage.measure import compare_ssim as ssim_metric
from scipy import signal
from scipy import ndimage
from PIL import Image, ImageDraw


from torchvision import datasets, transforms
from torch.autograd import Variable
import imageio


hostname = socket.gethostname()

def load_dataset(opt):
    """Load Moving MNIST dataset for SVG training."""
    from dataset.moving_mnist_svg import MovingMNISTSVG
    
    print(f"🎬 Loading Moving MNIST for SVG training:")
    print(f"   Data root: {opt.data_root}")
    print(f"   Train seq length: {opt.n_past + opt.n_future}")
    print(f"   Test seq length: {opt.n_eval}")
    print(f"   Image size: {opt.image_width}x{opt.image_width}")
    print(f"   Digits: {opt.num_digits}")
    
    train_data = MovingMNISTSVG(
        train=True,
        data_root=opt.data_root,
        seq_len=opt.n_past + opt.n_future,
        image_size=opt.image_width,
        deterministic=False,  # SVG uses stochastic movement
        num_digits=opt.num_digits
    )
    
    test_data = MovingMNISTSVG(
        train=False,
        data_root=opt.data_root,
        seq_len=opt.n_eval,
        image_size=opt.image_width,
        deterministic=False,  # SVG uses stochastic movement
        num_digits=opt.num_digits
    )
    
    print(f"✅ Datasets loaded:")
    print(f"   Train: {len(train_data)} samples")
    print(f"   Test: {len(test_data)} samples")
    
    return train_data, test_data

def sequence_input(seq, dtype):
    return [Variable(x.type(dtype)) for x in seq]

def normalize_data(opt, dtype, sequence):
    sequence.transpose_(0, 1)
    sequence.transpose_(3, 4).transpose_(2, 3)
    return sequence_input(sequence, dtype)

def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))

def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding :
                   (i+1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding :
                   (i+1) * y_dim + i * padding].copy_(image)
        return result

def save_np_img(fname, x):
    if x.shape[0] == 1:
        x = np.tile(x, (3, 1, 1))
    
    # Use our scipy.misc replacement
    img = scipy.misc.toimage(x, high=255*x.max(), channel_axis=0)
    img.save(fname)

def make_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    
    # Convert to PIL Image (scipy.misc.toimage replacement)
    arr = tensor.numpy()
    if hasattr(scipy.misc, 'toimage'):
        # Old scipy version
        return scipy.misc.toimage(arr, high=255*arr.max(), channel_axis=0)
    else:
        # New version - use our replacement
        return scipy.misc.toimage(arr, high=255*arr.max(), channel_axis=0)

def draw_text_tensor(tensor, text):
    np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
    pil = Image.fromarray(np.uint8(np_x*255))
    draw = ImageDraw.Draw(pil)
    draw.text((4, 64), text, (0,0,0))
    img = np.asarray(pil)
    return Variable(torch.Tensor(img / 255.)).transpose(1, 2).transpose(0, 1)

def save_gif(filename, inputs, duration=0.25):
    images = []
    for tensor in inputs:
        img = image_tensor(tensor, padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1)
        
        # Convert to numpy and ensure proper format for imageio
        img_np = img.numpy()
        
        # Handle different image formats
        if img_np.ndim == 3:
            if img_np.shape[2] == 1:
                # Single channel - convert to grayscale
                img_np = img_np.squeeze(axis=2)
            elif img_np.shape[2] == 3:
                # RGB - keep as is
                pass
        
        # Convert to uint8 range [0, 255]
        img_np = (img_np * 255).astype('uint8')
        
        images.append(img_np)
    
    imageio.mimsave(filename, images, duration=duration)

def save_gif_with_text(filename, inputs, text, duration=0.25):
    images = []
    for tensor, text in zip(inputs, text):
        img = image_tensor([draw_text_tensor(ti, texti) for ti, texti in zip(tensor, text)], padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1)
        
        # Convert to proper format for imageio
        img_np = img.numpy()
        
        # Handle image format
        if img_np.ndim == 3 and img_np.shape[2] == 1:
            img_np = img_np.squeeze(axis=2)
        
        # Convert to uint8
        img_np = (img_np * 255).astype('uint8')
        
        images.append(img_np)
    
    imageio.mimsave(filename, images, duration=duration)

def save_image(filename, tensor):
    img = make_image(tensor)
    img.save(filename)

def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return save_image(filename, images)

def prod(l):
    return functools.reduce(lambda x, y: x * y, l)

def batch_flatten(x):
    return x.resize(x.size(0), prod(x.size()[1:]))

def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            for c in range(gt[t][i].shape[0]):
                ssim[i, t] += ssim_metric(gt[t][i][c], pred[t][i][c])
                psnr[i, t] += psnr_metric(gt[t][i][c], pred[t][i][c])
            ssim[i, t] /= gt[t][i].shape[0]
            psnr[i, t] /= gt[t][i].shape[0]
            mse[i, t] = mse_metric(gt[t][i], pred[t][i])

    return mse, ssim, psnr

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            for c in range(gt[t][i].shape[0]):
                res = finn_ssim(gt[t][i][c], pred[t][i][c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(gt[t][i][c], pred[t][i][c])
            ssim[i, t] /= gt[t][i].shape[0]
            psnr[i, t] /= gt[t][i].shape[0]
            mse[i, t] = mse_metric(gt[t][i], pred[t][i])

    return mse, ssim, psnr


def finn_psnr(x, y):
    mse = ((x - y)**2).mean()
    return 10*np.log(1/mse)/np.log(10)


def gaussian2(size, sigma):
    A = 1/(2.0*np.pi*sigma**2)
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = A*np.exp(-((x**2/(2.0*sigma**2))+(y**2/(2.0*sigma**2))))
    return g

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()
  
def finn_ssim(img1, img2, cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 1 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

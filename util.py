from PIL import Image

def downsample(img, new_dims):
    """
    down sample img
    :param img: np.array img
    :param new_dims: (n, n) tuple of downsample target
    :return: downsampled img
    """
    # convert to pil Image
    img = Image.fromarray(img)
    # convert img
    img = img.resize(new_dims)
    return img

def preprocess(img, lr_dim, upscale_factor):
    """
    preprocess an image to be lr hr pair for a given dim
    :param img: full size img
    :param lr_dim: dims for low res
    :param upscale_factor: upscale factor for hr
    :return: lr hr pair
    """
    low_res = downsample(img, lr_dim)
    high_res = downsample(img, lr_dim*upscale_factor)
    return low_res, high_res
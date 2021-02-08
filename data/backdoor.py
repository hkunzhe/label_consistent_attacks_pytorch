import numpy as np
from PIL import Image


class CLBD(object):
    """ Label-Consistent Backdoor Attacks.

    Reference:
    [1] "Label-consistent backdoor attacks."
    Turner, Alexander, et al. arXiv 2019.

    Args:
        trigger_path (str): Trigger path.
    """

    def __init__(self, trigger_path):
        with open(trigger_path, "rb") as f:
            trigger_ptn = Image.open(f).convert("RGB")
        self.trigger_ptn = np.array(trigger_ptn)
        self.trigger_loc = np.nonzero(self.trigger_ptn)

    def __call__(self, img):
        return self.add_trigger(img)

    def add_trigger(self, img):
        """Add `trigger_ptn` to `img`.

        Args:
            img (numpy.ndarray): Input image (HWC).
        
        Returns:
            poison_img (np.ndarray): Poison image (HWC).
        """
        img[self.trigger_loc] = 0
        poison_img = img + self.trigger_ptn

        return poison_img

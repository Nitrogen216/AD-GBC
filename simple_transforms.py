import random

import numpy as np
from PIL import Image


class SimpleSegTransform:
    def __init__(self, height, width, training=False):
        self.height = height
        self.width = width
        self.training = training

    def __call__(self, image, mask):
        if self.training:
            k = random.randint(0, 3)
            if k:
                image = np.rot90(image, k)
                mask = np.rot90(mask, k)
            if random.random() < 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)

        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        image = np.array(
            Image.fromarray(image).resize((self.width, self.height), Image.BILINEAR)
        )

        resized_masks = []
        for c in range(mask.shape[2]):
            resized = Image.fromarray(mask[..., c]).resize(
                (self.width, self.height), Image.NEAREST
            )
            resized_masks.append(np.array(resized)[..., None])
        mask = np.concatenate(resized_masks, axis=2)

        return {"image": image, "mask": mask}

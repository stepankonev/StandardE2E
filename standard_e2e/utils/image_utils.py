import albumentations as A


class CropTop(A.ImageOnlyTransform):
    """Albumentations transform that removes a fraction from the top of images."""

    def __init__(self, top_cut_frac=0.0, always_apply=True):
        super().__init__(always_apply)
        self._top_cut_frac = top_cut_frac

    def apply(self, img, **params):  # pylint: disable=arguments-differ
        if self._top_cut_frac > 0:
            top_cut = int(img.shape[0] * self._top_cut_frac)
            return img[top_cut:, :]
        return img

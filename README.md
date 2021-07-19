### multi-organ nuclei instance segmentation and classification in histopathological images


Implementaion of the scale-adaptive feature selection U-Net with attention gate (AG), bottleneck attention module (BAM), convolutional block attention module (CBAM), squeeze-and-exitation (SE) and selective-kernel (SK).

AG: https://arxiv.org/abs/1808.08114

BAM: https://arxiv.org/abs/1807.06514

CBAM: https://arxiv.org/abs/1807.06521

SE: https://arxiv.org/abs/1709.01507?spm=a2c41.13233144.0.0

SK: https://arxiv.org/abs/1903.06586

Support multi-scale prediction, shortcut connection, dynamic feature selection.

Support cross-entropy loss, focal loss, strutural similarity (SSIM) loss, Dice loss and IoU loss.

Tested on datasets of the MoNuSeg 2018 https://monuseg.grand-challenge.org/Data/ and MoNuSAC 2020 https://monusac-2020.grand-challenge.org/Data/.

## MoNuSeg
![2019](/results/monuseg1.png)

![2020](/results/monuseg2.png)

## MoNuSAC
...


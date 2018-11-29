#http://scikit-image.org/docs/dev/auto_examples/transform/plot_piecewise_affine.html#sphx-glr-auto-examples-transform-plot-piecewise-affine-py

import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data

image = data.imread('initial_image.png')
rows, cols = image.shape[0], image.shape[1]
# some magic needed by scikit to construct an arbitray piecwise affine transformation
src_cols = np.linspace(0, cols, 100)
src_rows = np.linspace(0, rows, 2)
src_rows, src_cols = np.meshgrid(src_rows, src_cols)
src = np.dstack([src_cols.flat, src_rows.flat])[0]

maxiteration=10

for iteration in range(0, maxiteration):
    print("doing iteration ",iteration+1," out of ",maxiteration)
    # add superposition of 2 periodic oscillation to row with random amplitudes
    s=random.randint(20,50);
    dst_rows = src[:, 1] + (1.0+np.sin( np.linspace(0, 10 * np.pi, src.shape[0])) ) * s + (1.0+np.cos( np.linspace(0, 12 * np.pi, src.shape[0])) ) * s;
    dst_cols = src[:, 0]
    dst_rows *= 0.9 # to keep the image in view
    dst = np.vstack([dst_cols, dst_rows]).T
    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst )
    # perform wave distortion
    stage1 = warp(image, tform)
    # construct affine transformation with value a, the parameter -rows*a/2 shifts the result back to center
    a=random.randint(-100,100)/150.0;
    matrix = np.array([[1,a,  -rows*a/2], [0, 1,0], [ 0, 0, 1 ]])
    # perform affine transformation
    stage2 = warp(stage1, matrix)
    f, (p1, p2, p3) = plt.subplots(1, 3,sharey=True)
    p1.imshow(image)
    p2.imshow(stage1)
    p3.imshow(stage2)
    # rinse and repeat
    image=stage2
plt.show()
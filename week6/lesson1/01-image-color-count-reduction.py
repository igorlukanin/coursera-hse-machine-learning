from __future__ import print_function
from skimage.io import imread, imsave
from skimage import img_as_float
from sklearn.cluster import KMeans
from numpy import mean, median, reshape
from math import log10


def process_pixels(X, y, func):
    clusters = {}
    colors = {}

    for i in range(0, len(X)):
        if not clusters.has_key(y[i]):
            clusters[y[i]] = []

        clusters[y[i]].append(X[i])

    for i in range(0, len(clusters)):
        r = func(map(lambda x: x[0], clusters[i]))
        g = func(map(lambda x: x[1], clusters[i]))
        b = func(map(lambda x: x[2], clusters[i]))

        colors[i] = [r, g, b]

    return map(lambda x: colors[x], y)

def save_image(name, pixels, original):
    image = reshape(pixels, original.shape)
    imsave(name, image)

def square(I_pixel, K_pixel, i):
    delta = I_pixel[i] - K_pixel[i]
    return delta * delta

def MSE_RGB(I, K):
    deltas = []

    for i in range(0, len(I)):
        deltas.append(square(I[i], K[i], 0) + square(I[i], K[i], 1) + square(I[i], K[i], 2))

    return sum(deltas) / len(I) / 3

def PSNR(MSE):
    return - 10 * log10(MSE)


image = img_as_float(imread('parrots.jpg'))
X = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))

for j in range(1, 20 + 1):
    c = KMeans(n_clusters=j, init='k-means++', random_state=241)
    y = c.fit_predict(X)

    X_mean = process_pixels(X, y, mean)
    X_median = process_pixels(X, y, median)

    # save_image("parrots-mean-" + str(j) + ".jpg", X_mean, image)
    # save_image("parrots-median-" + str(j) + ".jpg", X_median, image)

    mean_MSE = MSE_RGB(X, X_mean)
    median_MSE = MSE_RGB(X, X_median)

    mean_PSNR = PSNR(mean_MSE)
    median_PSNR = PSNR(median_MSE)

    print()
    print(j, "clusters:")
    print("  Mean:    MSE=", '{0:.4f}'.format(mean_MSE), " PSNR=", '{0:.2f}'.format(mean_PSNR))
    print("  Median:  MSE=", '{0:.4f}'.format(median_MSE), " PSNR=", '{0:.2f}'.format(median_PSNR))


file = open('01-result.txt', 'w')
print('11', file=file, sep='', end='')
file.close()
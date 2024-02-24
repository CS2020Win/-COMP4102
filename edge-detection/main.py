import cv2
import numpy as np


def non_maximum_suppression(gradient: np.ndarray, theta: np.ndarray) -> np.ndarray:
    assert gradient.ndim == 2
    assert gradient.shape == theta.shape
    gradient = gradient.copy()
    for i in range(1, gradient.shape[0] - 1):
        for j in range(1, gradient.shape[1] - 1):
            if theta[i, j] < -3 * np.pi / 8 or theta[i, j] > 3 * np.pi / 8:
                if gradient[i, j] < max(gradient[i, j - 1], gradient[i, j + 1]):
                    gradient[i, j] = 0
            elif theta[i, j] < -np.pi / 8:
                if gradient[i, j] < max(gradient[i - 1, j + 1], gradient[i + 1, j - 1]):
                    gradient[i, j] = 0
            elif theta[i, j] < np.pi / 8:
                if gradient[i, j] < max(gradient[i - 1, j], gradient[i + 1, j]):
                    gradient[i, j] = 0
            else:
                if gradient[i, j] < max(gradient[i - 1, j - 1], gradient[i + 1, j + 1]):
                    gradient[i, j] = 0
    return gradient


def apply_gaussian_smoothing(img: np.ndarray, sigma: float) -> np.ndarray:
    hsize: int = int(2 * np.ceil(3 * sigma) + 1)
    gaussian = cv2.getGaussianKernel(hsize, sigma)
    gaussian = gaussian * gaussian.T
    return cv2.filter2D(img, -1, gaussian)


def estimate_gradient(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img = np.float32(img)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    imgx = cv2.filter2D(img, -1, sobel_x)
    imgy = cv2.filter2D(img, -1, sobel_y)

    gradient = np.sqrt(imgx**2 + imgy**2)
    theta = np.arctan(imgy, imgx)

    return gradient, theta


def apply_edge_detection(img: np.ndarray, sigma: float) -> np.ndarray:
    img = apply_gaussian_smoothing(img, sigma)
    gradient, theta = estimate_gradient(img)
    gradient = non_maximum_suppression(gradient, theta)
    return gradient


if __name__ == "__main__":
    import os

    data_dir = "../data"
    for filename in os.listdir(data_dir):
        img = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)
        img = apply_edge_detection(img, 1.0)
        gradient, theta = estimate_gradient(img)

        cv2.imwrite(
            f"gradient-magnitude-{filename}",
            cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
        )
        cv2.imwrite(
            f"gradient-orientation-{filename}",
            cv2.normalize(theta, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
        )

        edge = non_maximum_suppression(gradient, theta)
        cv2.imwrite(
            f"edge-detected-{filename}",
            cv2.normalize(edge, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
        )

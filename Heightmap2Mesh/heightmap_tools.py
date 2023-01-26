import numpy as np

def compute_slopes(heightmap):
    nx,ny = np.gradient(heightmap)
    slope_x = np.arctan2(nx,1)
    slope_y = np.arctan2(ny,1)
    magnitude = np.hypot(nx,ny)
    slope_xy = np.arctan2(magnitude,1)
    return slope_x, slope_y, slope_xy, magnitude



if __name__ == "__main__":
    X = np.load("/home/antoine/Documents/Moon/IsaacSimulationFramework/test_dem.npy")

    from matplotlib import pyplot as plt

    slope_x, slope_y, slope_xy, magnitude = compute_slopes(X)

    plt.figure()
    plt.imshow(slope_x)
    plt.figure()
    plt.imshow(slope_y)
    plt.figure()
    plt.imshow(slope_xy)
    plt.figure()
    plt.imshow(magnitude)
    plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from data_simulator.synthetic_data_creator import DataGenerator
import argparse

VIZ_PATH = 'data/viz'

def check_and_create_directory(dir_path: str) -> None:
    """Check if a directory exists, and if not, create it.
    :param dir_path: Path to the directory to check (str).
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")

def plot_alt_in_ranges(altitudes: np.ndarray, ranges: np.ndarray) -> None:
    """Plot altitude versus ranges and save the plot.
    :param altitudes: Array of altitudes (np.ndarray).
    :param ranges: Array of ranges corresponding to altitudes (np.ndarray).
    """
    plt.rcParams["figure.figsize"] = [14.00, 8.00]
    for i in range(len(altitudes)):
        plt.plot(ranges[i], altitudes[i], label=f'fragment {i + 1}')
    plt.xlabel("Downrange (m)")
    plt.ylabel("Altitude (m)")
    plt.legend()
    plt.grid()
    plt.savefig(f'{VIZ_PATH}/alt_in_ranges.jpg')

def plot_alt_time(altitudes: np.ndarray) -> None:
    """Plot altitude over time and save the plot.
    :param altitudes: Array of altitudes over time (np.ndarray).
    """
    plt.rcParams["figure.figsize"] = [30.00, 16.00]
    time = [i * 0.05 for i in range(len(altitudes[0]))]
    for i in range(len(altitudes)):
        plt.plot(time, altitudes[i], label=f'fragment {i + 1}')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Altitude (m)")
    plt.legend()
    plt.grid()
    plt.savefig(f'{VIZ_PATH}/alt_in_time.jpg')

def plot_carthesian(carthesian: np.ndarray) -> None:
    """Plot 3D Cartesian coordinates of data points.
    :param carthesian: Array of 3D Cartesian coordinates (np.ndarray).
    """
    fig = px.scatter_3d(x=carthesian[:, 0], y=carthesian[:, 1], z=carthesian[:, 2],
                        color_discrete_sequence=['red', 'green', 'blue'],
                        height=500, width=1000)
    fig.update_traces(marker=dict(size=3, line=dict(color='black', width=0.1)))
    fig.update_layout(title='Simulated data trajectory in Cartesian coordinate system')
    fig.write_image(f'{VIZ_PATH}/carthesian.jpg')
    fig.write_html(f'{VIZ_PATH}/carthesian.html')

def plot_alt_az(alt1: np.ndarray, alt2: np.ndarray, az1: np.ndarray, az2: np.ndarray) -> None:
    """Plot altitude and azimuth data in a polar coordinate system.
    :param alt1: Array of altitudes for camera 1 (np.ndarray).
    :param alt2: Array of altitudes for camera 2 (np.ndarray).
    :param az1: Array of azimuths for camera 1 (np.ndarray).
    :param az2: Array of azimuths for camera 2 (np.ndarray).
    """
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    theta_1, theta_2 = np.radians(az1), np.radians(az2)
    ax.scatter(theta_1, alt1, color='green', label='camera 1')
    ax.scatter(theta_2, alt2, color='blue', label='camera 2')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('Alt-Az Plot synthetic data of simulated debris reentry detected on camera')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.2))
    plt.savefig(f'{VIZ_PATH}/alt_az.jpg')

def plot_figs(altaz1: list, altaz2: list, altitudes: np.ndarray, ranges: np.ndarray, carthesian: np.ndarray) -> None:
    """Generate and save all plots.
    :param altaz1: Altitude and azimuth data for camera 1 (list of np.ndarray).
    :param altaz2: Altitude and azimuth data for camera 2 (list of np.ndarray).
    :param altitudes: Array of altitudes over time (np.ndarray).
    :param ranges: Array of ranges corresponding to altitudes (np.ndarray).
    :param carthesian: Array of 3D Cartesian coordinates (np.ndarray).
    """
    check_and_create_directory(f'{VIZ_PATH}')
    plot_alt_time(altitudes)
    plot_alt_in_ranges(altitudes, ranges)
    plot_carthesian(carthesian)
    plot_alt_az(altaz1[0], altaz2[0], altaz1[1], altaz2[1])

def main() -> None:
    """Main function to handle the operation of the script."""

    parser = argparse.ArgumentParser(description='Data Generator for Debris Simulation')
    parser.add_argument('--viz', type=bool,nargs=1, default=True, help='if first dataset data should be visualise (if yes results are stored in folder/viz)')


    args = parser.parse_args()

    dataGenerator = DataGenerator(num_sets=1,
                                   range_of_fragment_spread=[0.01, 5],
                                   angle=[80,126],
                                   debris_size=[0.01, 1.0],
                                   velocity=[500,7500],
                                   camera_fps=0.05,
                                   focal_length=16,
                                   pixel_size=4.65,
                                   num_of_frames=20,
                                   station1_coord_lon_lat=None,
                                   station2_coord_ra_dec=None,
                                   start_obst_time=None,
                                   mass=[10,250],
                                   drag_coef=30,
                                   surface_area=[25,100],
                                   lift_coef=[0, 0.01],
                                   num_of_fragments=15)
    data, alt2, range2 = dataGenerator.generate_data(get_3d=True)

    check_and_create_directory('data')
    with open(f'data/colab.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open(f'data/alr.pkl', 'wb') as f:
        pickle.dump(alt2, f)
    with open(f'data/ranges.pkl', 'wb') as f:
        pickle.dump(range2, f)

    if args.viz is True:
        plot_figs(altaz1=[data[0]['keypoints0'][0][:,0], data[0]['keypoints0'][0][:,1]], altaz2=[data[0]['keypoints1'][0][:,0], data[0]['keypoints1'][0][:,1]], 
                  altitudes=alt2[0], ranges=range2[0], carthesian=data[0]['carthesian'])
        
    print('Data saved')

if __name__ == '__main__':
    main()

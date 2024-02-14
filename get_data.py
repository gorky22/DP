### example of calling this code python3 get_data.py --number_of_datasets 20 --velocity 5000 7500 --lift_coef 0 0.000001 --angle 80 120  

from data_simulator.synthetic_data_creator import DataGenerator
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
import matplotlib.pyplot as plt
import random
import pickle as pkl
import argparse
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
import matplotlib.pyplot as plt


import os

def check_and_create_directory(dir_path):
    """
    Checks if a directory exists, and if it doesn't, creates it.

    :param dir_path: The path of the directory to check and create.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")


def plot_alt_in_ranges(altitudes, ranges):
    # Assuming 'data' is already defined and loaded as a list of 15 lists of altitudes
    altitudes = np.array(altitudes)
    ranges = np.array(ranges)

    plt.rcParams["figure.figsize"] = [14.00, 8.00]

    # Modify this line to scale the time values to seconds
    # Each index i is multiplied by 0.05 to convert it to seconds




    # Loop through each list of altitudes and plot them with different colors
    for i in range(len(altitudes)):
        plt.plot(ranges[i], altitudes[i],  label=f'fragment {i + 1}')

    plt.xlabel("Downrange (m)")
    plt.ylabel("Altitude (m)")
    plt.legend()
    plt.grid()
    plt.savefig('viz/alt_in_ranges.jpg')

def plot_alt_time(altitudes):
    altitudes = np.array(altitudes)

    plt.rcParams["figure.figsize"] = [30.00, 16.00]

    # Modify this line to scale the time values to seconds
    # Each index i is multiplied by 0.05 to convert it to seconds
    time = [i * 0.05 for i in range(len(altitudes[0]))]



    # Loop through each list of altitudes and plot them with different colors
    for i in range(len(altitudes)):
        plt.plot(time, altitudes[i], label=f'fragment {i + 1}')

    plt.xlabel("Time (seconds)")
    plt.ylabel("Altitude (m)")
    plt.legend()
    plt.grid()
    plt.savefig('viz/alt_in_time.jpg')

def plot_carthesian(carthesian):


    ax  = []
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #data[0]['3d'].append(station1_coord)

    data = np.array(carthesian)
    x= data[:,0]
    y = data[:,1]
    z = data[:,2]

    x = x[~np.isinf(x)]
    y = y[~np.isinf(y)]

    z = z[~np.isinf(z)]

    # Convert km to m (multiply by 1000)

    x = x 
    y = y 
    z = z

    colors = px.colors.sequential.Plasma
    colors[0], colors[1], colors[2] = ['red', 'green', 'blue']
    fig = px.scatter_3d(x=x, y=y, z=z, color_discrete_sequence=colors, height=500, width=1000)
    fig.update_traces(marker=dict(size=3, line=dict(color='black', width=0.1)))

    # set the layout
    fig.update_layout(scene=dict(
                xaxis=dict(backgroundcolor='white',
                            title='x [km]',
                            color='black',
                            gridcolor='#f0f0f0',
                            title_font=dict(size=10),
                            tickformat=',.0f', 
                            tickfont=dict(size=10)),
                yaxis=dict(backgroundcolor='white',
                            color='black',
                            title='y [km]',
                            gridcolor='#f0f0f0',
                            title_font=dict(size=10),
                            tickformat=',.0f', 
                            tickfont=dict(size=10)),
                zaxis=dict(backgroundcolor='lightgrey',
                            color='black', 
                            gridcolor='#f0f0f0',
                            title='z [km]',
                            title_font=dict(size=10),
                            tickfont=dict(size=10),
                            tickformat=',.0f')),
                showlegend=False,
                margin=dict(l=0, r=0, b=0, t=0))




    fig.update_layout(title='Simulated data trajectory in Cartesian coordinate system')

    fig.write_image('viz/carthesian.jpg')
    fig.write_html('viz/carthesian.html')

def plot_alt_az(alt1,alt2, az1, az2):
    # Generating another set of 1500 random data points for altitude and azimuth for the second dataset
    altitudes = alt1
    altitudes_2 = alt2

    # Create a polar subplot for the Alt-Az coordinate system
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

    # Convert azimuths to radians for the polar plot
    theta_1 = np.radians(az1)
    theta_2 = np.radians(az2)

    # Plot the first dataset in green
    ax.scatter(theta_1, altitudes, color='green', label='camera 1')

    # Plot the second dataset in blue
    ax.scatter(theta_2, altitudes_2, color='blue', label='camera 2')

    # Set the direction to be clockwise and start from the top
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Set labels and legend
    ax.set_title('Alt-Az Plot synthetic data of simulated debris reentry detected on camera')
    ax.set_rlabel_position(90)  # Position of the radial labels
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.2))

    # Display the plot
    plt.savefig('viz/alt_az.jpg')


def plot_figs(altaz1, altaz2, altitudes, ranges, carthesian):
    check_and_create_directory('viz')
    plot_alt_time(altitudes)
    plot_alt_in_ranges(altitudes, ranges)
    plot_carthesian(carthesian)
    plot_alt_az(altaz1[0], altaz2[0], altaz1[1], altaz2[1])


def parse_arguments():
    parser = argparse.ArgumentParser(description='Data Generator for Debris Simulation')

 
    parser.add_argument('--number_of_datasets', type=int,  default=1, help='how many datasets should generator generate')

    parser.add_argument('--range_of_fragment_spread', nargs=2, type=float, default=[0.01, 2],
                        help='Range for fragment spread (min max)')
    parser.add_argument('--angle', nargs=2, type=float, default=[14, 16], help='Angle range (min max)')
    parser.add_argument('--debris_size', nargs=2, type=float, default=[0.01, 1.0], help='Debris size range (min max)')
    parser.add_argument('--velocity', nargs=2, type=float, default=[7400, 7600], help='Velocity range (min max)')
    parser.add_argument('--camera_fps', type=float, default=0.05, help='Camera frames per second')
    parser.add_argument('--focal_length', type=int, default=16, help='Focal length of the camera')
    parser.add_argument('--pixel_size', type=float, default=4.65, help='Pixel size')
    parser.add_argument('--num_of_frames', type=int, default=1500, help='Number of frames')
    parser.add_argument('--mass', nargs=2, type=float, default=[20, 150], help='Mass range (min max)')
    parser.add_argument('--drag_coef', type=float, default=30, help='Drag coefficient')
    parser.add_argument('--surface_area', nargs=2, type=float, default=[25, 55], help='Surface area range (min max)')
    parser.add_argument('--lift_coef', nargs=2, type=float, default=[0, 0.1], help='Lift coefficient range (min max)')
    parser.add_argument('--num_of_fragments', type=int, default=15, help='Number of fragments')

    parser.add_argument('--viz', type=bool,nargs=1, default=True, help='if first dataset data should be visualise (if yes results are stored in folder/viz)')

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()

    if args.number_of_datasets >= 20:
        num_of_datasets = 20
    else:
        num_of_datasets = args.number_of_datasets

    counter = 1

    check_and_create_directory('data')
    while num_of_datasets <= args.number_of_datasets:

        dataGenerator = DataGenerator(num_of_datasets, 
                                   range_of_fragment_spread=args.range_of_fragment_spread, 
                                   angle=args.angle, 
                                   debris_size=args.debris_size, 
                                   velocity=args.velocity, 
                                   camera_fps=args.camera_fps, 
                                   focal_length=args.focal_length, 
                                   pixel_size=args.pixel_size, 
                                   num_of_frames=args.num_of_frames, 
                                   station1_coord_lon_lat=None, 
                                   station2_coord_ra_dec=None, 
                                   start_obst_time=None, 
                                   mass=args.mass, 
                                   drag_coef=args.drag_coef, 
                                   surface_area=args.surface_area, 
                                   lift_coef=args.lift_coef, 
                                   num_of_fragments=args.num_of_fragments)



        data,alt2,range2 = dataGenerator.generate_data(get_3d=True)

        
        with open(f'data/data_{counter}.pkl','wb') as f:
            pkl.dump(data, f)

            print(f'batch {counter} saved !')

        counter += 1
        num_of_datasets += 20
        


    if args.viz is True:
        plot_figs(altaz1=[data[0]['keypoints0'][0][:,0], data[0]['keypoints0'][0][:,1]], altaz2=[data[0]['keypoints1'][0][:,0], data[0]['keypoints1'][0][:,1]], 
                  altitudes=alt2[0], ranges=range2[0], carthesian=data[0]['carthesian'])

import random
import math
import numpy as np
from astropy.coordinates import *
import astropy.units as au
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation


class Fragment():

    def get_random_number_in_ranges(self, minimum: float, maximum: float)-> float:
        """
        Get descriptors within a specified range.

        :param min: Minimum value.
        :param max: Maximum value.

        :return: Normalized descriptor value.
        """

        number = random.uniform(minimum, maximum)

        return number
    

    def __init__(self, id_, location_of_fragment, angle=-0.1, debris_size=300, initial_altitude=100000, initial_velocity=7500, camera_fps=0.5, focal_length=16, pixel_size=4.65, num_of_frames=1000,mass=12, drag_coef=1.5,
                                       surface_area=0.3, lift_coef=1.0):
        # Constants
        self.GRAVITY_ACCEL = 9.81
        self.RADIUS_EARTH = 6371000

        # Instance Variables
        self.angle = [angle]
        self.debris_size = debris_size
        self.alt = [initial_altitude]
        self.velocity = [initial_velocity]
        self.camera_fps = camera_fps
        self.downrange = []
        self.mass = mass  # kg
        self.drag_coef = drag_coef
        self.surface_area = surface_area  # m^2
        self.lift_coef = lift_coef
        self.id_ = id_
        self.location_of_fragment = location_of_fragment
        self.num_of_frames = num_of_frames
        self.focal_length = focal_length
        self.pixel_size = pixel_size
        self.initial_intensity1 = self.get_random_number_in_ranges(1000000 * (20**2), 2000000 * (20**2) * 5)
        self.initial_intensity2 = self.get_random_number_in_ranges(1000000 * (20**2), 2000000 * (20**2) * 5)
        self.ambient_light_avg = 10
        self.ambient_light_variation = 3
        self.sensor_noise_floor = 15
        self.temperature_variation = 2

        self.final = {'keypoints0': [], 'scores0': [], 'descriptors0': [[], [], [], [], [], []], 'keypoints1': [], 'scores1': [], 'descriptors1': [[], [], [], [], [], []], 'fno': [], 'id_': [], 'carthesian': []}

    def altitude_from_geocentric(self, x_coordination: float, y_coordination: float, z_coordination: float) -> list:
        """
        Calculate altitude from geocentric coordinates.

        :param X: Geocentric X coordinate in kilometers.
        :param Y: Geocentric Y coordinate in kilometers.
        :param Z: Geocentric Z coordinate in kilometers.
        :return: Altitude in kilometers.
        """

        return math.sqrt(x_coordination**2 + y_coordination**2 + z_coordination**2) - self.RADIUS_EARTH

    def geocentric_from_altitude(self, x_coordination: float, y_coordination: float, new_altitude: float) -> list:
        """
        Calculate new Z coordinate from a new altitude.

        :param X: Original geocentric X coordinate in kilometers.
        :param Y: Original geocentric Y coordinate in kilometers.
        :param new_altitude: New altitude in kilometers.
        :return: New geocentric Z coordinate in kilometers.
        """

        outer_earth_radius = self.RADIUS_EARTH + new_altitude
        Z_new = math.sqrt(abs(outer_earth_radius **2 - x_coordination**2 - y_coordination**2))
        return [x_coordination, y_coordination, Z_new]

    def get_position_in_time(self, direction_vector: list, start_point: list, step: int) ->list:
        """
        Calculate the object's position at a specific time step.

        :param direction_vector: Direction vector indicating object's movement direction.
        :param start_point: Initial position of the object (X, Y, Z in km).
        :param step: Number of time steps to simulate.
        :return: New position (X, Y, Z in km) of the object after the given time steps.
        """

        weight = self.mass * self.GRAVITY_ACCEL
        lift_to_drag_ratio = self.lift_coef / self.drag_coef
        ballistic_coef = weight / (self.drag_coef * self.surface_area)

        curr_alt = self.altitude_from_geocentric(*start_point) * 1000
        curr_vel = self.velocity[-1]
        curr_angle = self.angle[-1]
        downrange = 0

        self.alt = []
        self.downrange = []

        for _ in range(step):

            self.alt.append(curr_alt)
            if curr_alt > 25000:
                T = -131.21 + (0.00299 * curr_alt)
                P = 2.488 * (((T + 273.1) / 216.6) ** -11.388)
            elif 11000 <= curr_alt <= 25000:
                T = -56.46
                P = 22.65 * math.exp(1.73 - (0.000157 * curr_alt))
            elif 0 < curr_alt < 11000 :
                T = 15.04 - (0.00649 * curr_alt)
                P = 101.29 * (((T + 273.1) / 288.08) ** 5.256)
            else:
                return start_point

            # Update velocity and position
            p = P / (0.2869 * (T + 273.1))
            Q = p * curr_vel**2 / 2
            curr_vel += self.camera_fps * self.GRAVITY_ACCEL * ((-Q / ballistic_coef) + math.sin(curr_angle))
            curr_angle += self.camera_fps * (((-(Q * self.GRAVITY_ACCEL) / ballistic_coef) * lift_to_drag_ratio +
                                 (math.cos(curr_angle) * (self.GRAVITY_ACCEL - ((curr_vel ** 2) / (self.RADIUS_EARTH + curr_alt))))) / curr_vel)
            curr_alt += self.camera_fps * (-curr_vel) * math.sin(curr_angle)
            downrange = self.camera_fps * ((self.RADIUS_EARTH * curr_vel * math.cos(curr_angle)) / (self.RADIUS_EARTH + curr_alt))

            # Update position

            self.downrange.append(downrange)
            displacement_vector = [curr_vel/1000 * self.camera_fps * x for x in direction_vector]
            start_point = [start_point[i] + displacement_vector[i] for i in range(2)]

        return self.geocentric_from_altitude(start_point[0], start_point[1], curr_alt/1000)



    def get_altaz(self, points: list, observatory: list, obst_times: str) -> list:
        """
        Convert geocentric Cartesian coordinates to Altitude-Azimuth coordinates.

        :param points: Array of geocentric Cartesian coordinates (Nx3).
        :param observatory: Observatory coordinates (latitude, longitude, height).
        :param obst_times: Array of observation times.

        :return: Array of Altitude-Azimuth coordinates (Az, Alt).
        """

        points = np.array(points)



        # Create SkyCoord objects for the object and the camera
        skycoord_position = SkyCoord(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            unit='km',
            representation_type='cartesian',
            frame='gcrs',
            obstime=Time(obst_times)
        )

        # Create the EarthLocation object.
        earth_location = EarthLocation(
            lat=observatory['lat'],
            lon=observatory['lon'],
            height=observatory['h']*au.m,

        )

        altaz_position = skycoord_position.transform_to(AltAz(obstime=Time(obst_times), location=earth_location))

        return np.column_stack((altaz_position.az.deg, altaz_position.alt.deg))



    def norm(self, val: float, min: float, max: float)-> float:
        """
        Normalize a value within a given range.

        :param val: Value to normalize.
        :param min: Minimum value of the range.
        :param max: Maximum value of the range.

        :return: Normalized value.
        """
        return (val - min) / (max - min)




    def compute_intensity(self,  initial_intensity: float,distance: float) -> float:
        """
        Compute intensity based on the inverse-square law.

        :param distance: Distance from the source.
        :param initial_intensity: Initial intensity.

        :return: Computed intensity.
        """

        intensity = initial_intensity / (distance ** 2)
        return intensity


    def calculate_gaussian_width(self, meteor_size: float, camera_focal_length: float, meteor_distance: float, pixel_size: float) -> float:
        """
        Convert pixel size to meters from micrometers

        :return: Computed gausian width.
        """
        
        pixel_size_meters = pixel_size * 1e-6

        gaussian_width_meters = (meteor_size * camera_focal_length) / meteor_distance

        gaussian_width_pixels = gaussian_width_meters / pixel_size_meters
        return gaussian_width_pixels


    def dynamic_gaussian_baseline(self) -> float:
        """
        Simulate Gaussian baseline dynamically with changing ambient light and sensor noise due to temperature.

        :return: Simulated Gaussian baseline.
        """

        gaussian_baseline = 23 + random.uniform(-2, 2)

        return gaussian_baseline

    def gaussian_peak_height_from_area(self, area: float, sigma: float) -> float:
        """
        Calculate the height of a Gaussian peak given the area under the curve and the standard deviation.

        :param area: Area under the Gaussian curve (float).
        :param sigma: The standard deviation of the Gaussian peak (float).
        :return: Height of the Gaussian peak (float).
        """
        height = area / (math.sqrt(2 * math.pi) * sigma)
        return height

    def get_final_dict(self, camera1_pos: tuple, camera2_pos: tuple, obst_times: list) -> dict:
        """
        Compile a final dictionary of keypoints for two camera positions based on Cartesian coordinates.

        :param camera1_pos: The position of camera 1 (tuple).
        :param camera2_pos: The position of camera 2 (tuple).
        :param obst_times: List of observation times (list).
        :return: A dictionary with keypoints for both cameras (dict).
        """
        self.final['keypoints0'].append(self.get_altaz(self.final['carthesian'], camera1_pos, obst_times))
        self.final['keypoints1'].append(self.get_altaz(self.final['carthesian'], camera2_pos, obst_times))
        return self.final

    def get_start_point(self, start_point: tuple) -> np.ndarray:
        """
        Calculate the start point of a fragment based on its initial location.

        :param start_point: The starting point coordinates (tuple).
        :return: Adjusted start point as a numpy array (np.ndarray).
        """
        return np.array(start_point) + self.location_of_fragment

    def simulate_meteor_data(self, intensity: float, meteor_size: float, camera_focal_length: float, meteor_distance: float, pixel_size: float) -> list:
        """
        Simulate meteor data based on various input parameters and calculate the characteristics of the observed Gaussian blur.

        :param intensity: The intensity of the meteor (float).
        :param meteor_size: The physical size of the meteor (float).
        :param camera_focal_length: The focal length of the camera lens (float).
        :param meteor_distance: The distance from the camera to the meteor (float).
        :param pixel_size: The size of the camera sensor's pixels (float).
        :return: A list containing the Gaussian height, width, baseline, and the number of oversaturated pixels (list).
        """
        gaussian_width = self.calculate_gaussian_width(meteor_size, camera_focal_length, meteor_distance, pixel_size)
        gaussian_height = self.gaussian_peak_height_from_area(intensity, gaussian_width)
        gaussian_baseline = self.dynamic_gaussian_baseline()
        intensity_threshold = 255
        num_oversaturated_pixels = 0
        if intensity > intensity_threshold:
            num_oversaturated_pixels = math.ceil(intensity / intensity_threshold)
        return [gaussian_height, gaussian_width, gaussian_baseline, num_oversaturated_pixels]



    def get_fragmet_data(self,start_point: list, frame_num: int, camera1_coord: list,camera2_coord: list, dir_vector: list)-> tuple:
        '''
            Generate synthetic data.

            :param get_3d: Boolean indicating whether to generate 3D data.

            :return: List of generated data.
        '''

        corected_start_point = self.get_start_point(start_point)

        kp = self.get_position_in_time(dir_vector, corected_start_point, frame_num)

        meteor_distance1 =  np.linalg.norm(np.array(kp) - np.array(camera1_coord))
        meteor_distance2 =  np.linalg.norm(np.array(kp) - np.array(camera2_coord))

        intensity1 = self.compute_intensity(self.initial_intensity1, meteor_distance1)
        intensity2 = self.compute_intensity(self.initial_intensity1, meteor_distance2)

        desc0 = self.simulate_meteor_data(intensity1, self.debris_size,self.focal_length, meteor_distance1 * 1000, self.pixel_size)
        desc1 = self.simulate_meteor_data(intensity2, self.debris_size, self.focal_length, meteor_distance2 * 1000,self.pixel_size)

        desc0.append(intensity1)
        desc1.append(intensity2)

        [self.final['descriptors1'][i].append(desc1[i]) for i in range(0,len(desc0))]
        [self.final['descriptors0'][i].append(desc0[i]) for i in range(0,len(desc0))]

        self.final['carthesian'].append(kp)
        self.final['fno'].append(frame_num)
        self.final['id_'].append(self.id_)
        self.final['scores0'].append(1)
        self.final['scores1'].append(1)

        return kp
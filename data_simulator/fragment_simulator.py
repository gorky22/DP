import random
import math
import datetime as dt
import os
import numpy as np
from celest.coordinates import GroundLocation as cGroundLocation
from celest.coordinates import Coordinate as cCoordinate
from celest.coordinates import GCRS as cGCRS
from celest import units as cu
from celest.coordinates import AzEl as cAzEl
import pandas as pd
from datetime import datetime, timedelta
from astropy.coordinates import *
import astropy.units as au
from astropy.time import Time
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time

#https://fmph.uniba.sk/fileadmin/fmfi/microsites/kafzm/daa/Metory/AMOS_technical.jpg
#https://stackoverflow.com/questions/50125574/calculate-image-size-of-an-object-from-real-size
# focal length of modular automatic system fisheye lens CANON/SIGMA Image intensigfier MULARD XX1332 image lens of shade of amplifier meopta video oprication 1.4/16 digital camera sourcing DMK41BU02F/CCD cameras with large chips
## 16mm
class Fragment():

    angle = []
    alt  = []
    range = []
    dt = []
    speed = []
    downrange = []

    num_of_frames = None
    pixel_size = None
    focal_length = None
    debris_size = None

         # Updated Constants and Initial Conditions for AMOS Detection
    mass = 12  # Mass of the reentry object in kg
    drag_coef = 1.5  # Drag coefficient
    surface_area = 0.3  # Surface area in square meters
    ift_coef = 1.0  # Lift coefficient
    location_of_fragment = None
    initial_intensity1 = None
    initial_intensity2 = None
    final = {'keypoints0': [], 'scores0':[], 'descriptors0':[[],[],[],[],[],[]],'keypoints1':[], 'scores1':[], 'descriptors1':[[],[],[],[],[],[]], 'fno':[],'id_':[],'carthesian': []}
    id_ = None
    descriptors = {'intensity_range' : {'min':1000000 * (20**2) , 'max': 2000000 * (20**2) * 5},    #####  lets say obj ma intesitu nejaku .... vo vzdialenosti v od objektu sa javi ze ma jasnost povodna / v na 2
                    'GaussianHeight_range' : {'min': 0, 'max': 1000},
                    'GaussianWidth_range' : {'min': 0, 'max': 10},
                    'GaussianBaseline_range' : {'min': 0, 'max': 100},
                    'NumOversaturatedPixels_range' : {'min': 0, 'max': 100}}
    
    def get_descriptors(self, min, max):
        """
        Get descriptors within a specified range.

        :param min: Minimum value.
        :param max: Maximum value.

        :return: Normalized descriptor value.
        """
        

        start_desc = random.uniform(min, max)

        #norm_start_desc = self.norm(start_desc, min, max)
        #print(min,max,norm_start_desc)

        return start_desc
    def __init__(self, id_, location_of_fragment, angle=-0.1, debris_size=300, initial_altitude=100000, initial_velocity=7500, camera_fps=0.5, focal_length=16, pixel_size = 4.65,  num_of_frames=1000, station1_coord_lon_lat=None, station2_coord_ra_dec=None, start_obst_time=None,      # Updated Constants and Initial Conditions for AMOS Detection
                                        mass = 12,  # Mass of the reentry object in kg
        drag_coef =1.5,  # Drag coefficient
        surface_area =0.3 , # Surface area in square meters
        lift_coef = 1.0  # Lift coefficient):
        ):
        """
        Initialize the DataGenerator object.

        :param num_sets: Number of sets to generate.
        :param station1_coord_lon_lat: Coordinates of station 1 (camera).
        :param station2_coord_ra_dec: Coordinates of station 2 (camera).
        :param start_obst_time: Starting observation time.
        """

             # Updated Constants and Initial Conditions for AMOS Detection
        self.mass = mass  # Mass of the reentry object in kg
        self.drag_coef = drag_coef  # Drag coefficient
        self.surface_area = surface_area  # Surface area in square meters
        self.lift_coef = lift_coef  # Lift coefficient
        self.id_ = id_

        self.location_of_fragment = location_of_fragment

        if station1_coord_lon_lat is not None:
            self.station_coord_lon_lat = station1_coord_lon_lat

        self.angle.append(angle)
        self.alt.append(initial_altitude)
        self.speed.append(initial_velocity)
        self.debris_size = debris_size
        self.initial_intensity1=self.get_descriptors(self.descriptors['intensity_range']['min'],self.descriptors['intensity_range']['max'])
        self.initial_intensity2= self.get_descriptors(self.descriptors['intensity_range']['min'],self.descriptors['intensity_range']['max'])
        
        self.dt = camera_fps
 
          
        self.num_of_frames = num_of_frames
             # Replace these with your actual values

        self.focal_length = focal_length  # Focal length in mm
            #meteor_distance = 80000.0  # Distance in meters
        self.pixel_size = pixel_size  # Pixel size in micrometers
 
        self.final = {'keypoints0': [], 'scores0':[], 'descriptors0':[[],[],[],[],[],[]],'keypoints1':[], 'scores1':[], 'descriptors1':[[],[],[],[],[],[]], 'fno':[],'id_':[],'carthesian': []}
        #self.station1_coord_ra_dec = self.get_staton_eloc(self.station1_coord_ra_dec[0], self.station1_coord_ra_dec[1],self.station1_coord_ra_dec[2]) 
        #self.station2_coord_ra_dec = self.get_staton_eloc(self.station2_coord_ra_dec[0], self.station2_coord_ra_dec[1],self.station2_coord_ra_dec[2])


    def get_position_in_time(self, direction_vector, start_point, step):
        """
        Calculate the object's position at a specific time step.

        :param direction_vector: Direction vector indicating object's movement direction.
        :param start_point: Initial position of the object (X, Y, Z in km).
        :param step: Number of time steps to simulate.
        :return: New position (X, Y, Z in km) of the object after the given time steps.
        """

        def altitude_from_geocentric(X, Y, Z):
            """
            Calculate altitude from geocentric coordinates.

            :param X: Geocentric X coordinate in kilometers.
            :param Y: Geocentric Y coordinate in kilometers.
            :param Z: Geocentric Z coordinate in kilometers.
            :return: Altitude in kilometers.
            """
            R_earth = 6371  # Earth's mean radius in kilometers
            return math.sqrt(X**2 + Y**2 + Z**2) - R_earth

        def geocentric_from_altitude(X, Y, new_altitude):
            """
            Calculate new Z coordinate from a new altitude.

            :param X: Original geocentric X coordinate in kilometers.
            :param Y: Original geocentric Y coordinate in kilometers.
            :param new_altitude: New altitude in kilometers.
            :return: New geocentric Z coordinate in kilometers.
            """
            R_earth = 6371  # Earth's mean radius in kilometers
            r_new = R_earth + new_altitude
            Z_new = math.sqrt(abs(r_new**2 - X**2 - Y**2))
            return [X, Y, Z_new]

        # Constants and initial conditions
        gravity_accel = 9.81  # Acceleration due to gravity in m/s^2
        radius_earth = 6371000  # Radius of the Earth in meters
        weight = self.mass * gravity_accel
        lift_to_drag_ratio = self.lift_coef / self.drag_coef
        ballistic_coef = weight / (self.drag_coef * self.surface_area)
        dt = 0.05  # Time step in seconds

        curr_alt = altitude_from_geocentric(*start_point) * 1000
        curr_vel = self.speed[-1]
        curr_angle = self.angle[-1]
        downrange = 0

        self.alt = []
        self.downrange = []
        for j in range(step):
            # Atmospheric conditions
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
            curr_vel += dt * gravity_accel * ((-Q / ballistic_coef) + math.sin(curr_angle))
            curr_angle += dt * (((-(Q * gravity_accel) / ballistic_coef) * lift_to_drag_ratio + 
                                 (math.cos(curr_angle) * (gravity_accel - ((curr_vel ** 2) / (radius_earth + curr_alt))))) / curr_vel)
            curr_alt += dt * (-curr_vel) * math.sin(curr_angle)
            downrange = dt * ((radius_earth * curr_vel * math.cos(curr_angle)) / (radius_earth + curr_alt))

            # Update position
            
            self.downrange.append(downrange)
            displacement_vector = [curr_vel/1000 * dt * x for x in direction_vector]
            start_point = [start_point[i] + displacement_vector[i] for i in range(2)]

        return geocentric_from_altitude(start_point[0], start_point[1], curr_alt/1000)



    def get_altaz(self, points, observatory, obst_times):
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

        # Convert to AltAz coordinates.
        altaz_position = skycoord_position.transform_to(AltAz(obstime=Time(obst_times), location=earth_location))

        

        #print(altaz_position.az, altaz_position.az.deg)

        return np.column_stack((altaz_position.az.deg, altaz_position.alt.deg))
        #return np.column_stack((altaz_position.az, altaz_position.alt))



    def norm(self, val, min, max):
        """
        Normalize a value within a given range.

        :param val: Value to normalize.
        :param min: Minimum value of the range.
        :param max: Maximum value of the range.

        :return: Normalized value.
        """
        return (val - min) / (max - min)



        
    def compute_intensity(self,  initial_intensity,distance):
        """
        Compute intensity based on the inverse-square law.

        :param distance: Distance from the source.
        :param initial_intensity: Initial intensity.

        :return: Computed intensity.
        """
    # Using the inverse-square law

        intensity = initial_intensity / (distance ** 2)
        return intensity


    def calculate_gaussian_width(self, meteor_size, camera_focal_length, meteor_distance, pixel_size):
        # Convert pixel size to meters from micrometers
        pixel_size_meters = pixel_size * 1e-6
        # Calculate the Gaussian width in meters
        gaussian_width_meters = (meteor_size * camera_focal_length) / meteor_distance
        # Convert the Gaussian width to pixels
        gaussian_width_pixels = gaussian_width_meters / pixel_size_meters
        return gaussian_width_pixels
    

    def dynamic_gaussian_baseline(self, ambient_light_avg, ambient_light_variation, sensor_noise_floor, temperature_variation):
        """
        Simulate Gaussian baseline dynamically with changing ambient light and sensor noise due to temperature.

        :param ambient_light_avg: The average level of ambient light.
        :param ambient_light_variation: The potential variation in ambient light.
        :param sensor_noise_floor: The baseline noise level of the sensor at a reference temperature.
        :param temperature_variation: The variation in sensor noise due to temperature changes.
        :return: Simulated Gaussian baseline.
        """
        # Simulate dynamic ambient light level
        ambient_light = ambient_light_avg + random.uniform(-ambient_light_variation, ambient_light_variation)

        # Simulate sensor noise change due to temperature
        sensor_noise = sensor_noise_floor + random.uniform(-temperature_variation, temperature_variation)

        gaussian_baseline = ambient_light + sensor_noise
        return gaussian_baseline


    
    def generate_point(self, camera1, random_distance=None, dir_vector=None):
            """
            Generate a random direction vector.

            :return: Random direction vector.
            """

            # Choose one of the cameras as the reference camera
            reference_camera = camera1  # You can choose camera1 or camera2
            
            was_none = False
        
            # Calculate the vector from the reference camera to the other camera
            if dir_vector is None:
                
                dir_vector = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.5, 1)]

            # Generate a random distance between 80 km and 120 km
            if random_distance is None:
                was_none = True
                random_distance = random.uniform(90, 110)


            # Calculate the coordinates of the random point
            point = (reference_camera[0] + random_distance * dir_vector[0], 
                            reference_camera[1] + random_distance * dir_vector[1], 
                            reference_camera[2] + random_distance * dir_vector[2])
            
    
            
            if was_none:

                return list(point), random_distance, dir_vector
            else:
                return list(point)
            
    def get_final_dict(self, camera1_pos, camera2_pos, obst_times):
        self.final['keypoints0'].append( self.get_altaz(self.final['carthesian'],camera1_pos,obst_times))
        self.final['keypoints1'].append( self.get_altaz(self.final['carthesian'],camera2_pos,obst_times))
        return self.final
            
    def get_start_point(self, start_point):
        return np.array(start_point) + self.location_of_fragment
    
    def simulate_meteor_data(self, intensity, meteor_size, camera_focal_length, meteor_distance, pixel_size):

     
        #https://mae.ufl.edu/~uhk/GAUSSIAN-NEW.pdf
        gaussian_width = self.calculate_gaussian_width(meteor_size, camera_focal_length, meteor_distance, pixel_size)

        # Calculate Gaussian Height based on the intensity (you can modify this as needed)
        


        def gaussian_peak_height_from_area(area, sigma):
            """
            Calculate the height of a Gaussian peak given the area under the curve and 2 times the standard deviation.

            :param area: Area under the Gaussian curve
            :param two_sigma: Two times the standard deviation of the Gaussian peak
            :return: Height of the Gaussian peak
            """

            #height = area * (2.35 / np.sqrt(2 * np.pi))
            #sigma = two_sigma / 2


            height = area / (math.sqrt(2 * math.pi) * sigma)
            return height
        
      
        gaussian_height = gaussian_peak_height_from_area(intensity, gaussian_width)
        # Gaussian Baseline (assuming a fixed value, modify as needed)

        # Example usage with hypothetical values
        ambient_light_avg = 10
        ambient_light_variation = 3
        sensor_noise_floor = 15
        temperature_variation = 2

        # Simulate over a sequence of frames
        gaussian_baseline = self.dynamic_gaussian_baseline(ambient_light_avg, ambient_light_variation, sensor_noise_floor, temperature_variation)

        # Determine oversaturated pixels (assuming 8-bit ADC resolution)
        intensity_threshold = 2 ** 8 - 1
        num_oversaturated_pixels = 0
        if intensity > intensity_threshold:
            num_oversaturated_pixels = math.ceil(intensity/ intensity_threshold)

    

        return [gaussian_height, gaussian_width,gaussian_baseline ,num_oversaturated_pixels]



    def get_fragmet_data(self,start_point, frame_num, camera1_coord,camera2_coord, dir_vector):
        '''
            Generate synthetic data.

            :param get_3d: Boolean indicating whether to generate 3D data.

            :return: List of generated data.
        '''


        #kp0, kp1 = self.get_altaz(self.get_position_in_time(points[-1], dir_vector))
        #print(start_point)
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


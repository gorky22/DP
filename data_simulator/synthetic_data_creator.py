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
from data_simulator.fragment_simulator import Fragment
from tqdm import tqdm
##https://mae.ufl.edu/~uhk/GAUSSIAN-NEW.pdf
#https://fmph.uniba.sk/fileadmin/fmfi/microsites/kafzm/daa/Metory/AMOS_technical.jpg
#https://stackoverflow.com/questions/50125574/calculate-image-size-of-an-object-from-real-size
# focal length of modular automatic system fisheye lens CANON/SIGMA Image intensigfier MULARD XX1332 image lens of shade of amplifier meopta video oprication 1.4/16 digital camera sourcing DMK41BU02F/CCD cameras with large chips
## 16mm
class DataGenerator():
    """
    This class is designed to generate data for simulating the observation of atmospheric objects, such as meteors or space debris, as they pass through the Earth's atmosphere. It uses a variety of parameters to simulate different scenarios for observation by ground-based stations.

    Attributes:
    - station1_coord_lon_lat: Longitude, latitude, and height of observation station 1.
    - station2_coord_lon_lat: Longitude, latitude, and height of observation station 2.
    - angle_range: Tuple indicating the range of angles of observation.
    - altitude_range: Tuple indicating the range of altitudes of the observed object.
    - velocity_range: Tuple indicating the range of velocities of the observed object.
    - dt: Time interval for the simulation.
    - speed: Speed of the object.
    - descriptors: Various descriptors for the observed object, like intensity, Gaussian properties, etc.
    - start_obst_time: Starting time for the observation.
    - amos_time_interval: Time interval specific to the AMOS (Astronomical Multimode Observational System) system.
    - object_in_atmosphere_range_speed: Range of speeds for the object while in the atmosphere.
    - num_of_frames: Number of frames for the simulation.
    - pixel_size: Size of the camera's pixels.
    - focal_length: Focal length of the camera.
    - debris_size: Size of the space debris.
    - mass, drag_coef, surface_area, lift_coef: Physical properties of the reentry object.
    """
    # Static attributes
    station2_coord_lon_lat = {'lon': -156.256146, 'lat': 20.707403, 'h': 3068}
    station1_coord_lon_lat = {'lon': -155.477173, 'lat': 19.823662, 'h': 4126}
    descriptors = {...}  # As previously defined
    start_obst_time = '2019-10-15 07:38:22.800'
    amos_time_interval = 0.05
    object_in_atmosphere_range_speed = [12, 40]

    # Added range attributes
    angle = (-10, 10)  # Example range, adjust as needed
    altitude_range = (80000, 120000)  # Example range in meters
    velocity = (5000, 10000)  # Example range in m/s

    # Other attributes
    num_of_frames = None
    pixel_size = None
    focal_length = None
    debris_size = None
    num_of_fragments = None
    range_of_fragment_spread = None

    def __init__(self, num_sets,range_of_fragment_spread, angle, velocity, debris_size, camera_fps, focal_length, pixel_size, num_of_frames, station1_coord_lon_lat=None, station2_coord_ra_dec=None, start_obst_time=None, mass=12, drag_coef=1.5, surface_area=0.3, lift_coef=1.0,num_of_fragments=1):
        """
        Initialize the DataGenerator object with dynamic or constant parameters for simulation.

        :param num_sets: Number of sets to generate.
        :param angle: Angle or range of angles for observation.
        :param altitude: Altitude or range of altitudes for the object.
        :param velocity: Velocity or range of velocities for the object.
        :param debris_size: Size of the debris.
        :param camera_fps: Frames per second of the camera.
        :param focal_length: Focal length of the camera.
        :param pixel_size: Size of the camera's pixels.
        :param num_of_frames: Number of frames for the simulation.
        :param station1_coord_lon_lat: Coordinates of station 1.
        :param station2_coord_ra_dec: Coordinates of station 2.
        :param start_obst_time: Starting observation time.
        :param mass: Mass of the reentry object.
        :param drag_coef: Drag coefficient.
        :param surface_area: Surface area of the object.
        :param lift_coef: Lift coefficient.
        """

        self.num_sets = num_sets
        self.mass = mass
        self.drag_coef = drag_coef
        self.surface_area = surface_area
        self.lift_coef = lift_coef
        self.angle = angle
        self.velocity = velocity
        self.range_of_fragment_spread = range_of_fragment_spread

               
        self.debris_size = debris_size
        self.dt = camera_fps
        self.num_of_frames = num_of_frames
        self.focal_length = focal_length
        self.pixel_size = pixel_size
        self.num_of_fragments = num_of_fragments
        

        if station1_coord_lon_lat:
            self.station1_coord_lon_lat = station1_coord_lon_lat
        else:
            self.station1_coord_lon_lat = self.station1_coord_lon_lat

        if station2_coord_ra_dec:
            self.station2_coord_ra_dec = station2_coord_ra_dec
        else:
            self.station2_coord_ra_dec = self.station2_coord_lon_lat

        self.start_obst_time = start_obst_time if start_obst_time else self.start_obst_time


    def add_mls_to_obs_time(self, original_time, to_add):
        """
        Add milliseconds to the original timestamp.

        :param original_time: Original timestamp.
        :param to_add: Milliseconds to add.

        :return: New timestamp.
        """
     
        # Add 0.05 seconds
        new_time = original_time + timedelta(seconds=to_add)

 
        return new_time
    

    def get_staton_eloc(self, lon, lat, height):
        '''
        _get_staton_eloc function takes coordinates of station (camera) and returns Earthlocation object (astropy)
        :param lon: longtitude  
        :param lat: latitude
        :param height: height (in meters)
        :return: returns Earthlocation object
        '''

        
        return EarthLocation(lon=lon*au.deg, lat=lat*au.deg, height=height*au.m) 

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
            
            dir_vector = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.9,1)]
            dir_vector = np.array(dir_vector)
            norm = np.linalg.norm(dir_vector)
            dir_vector = dir_vector / norm

        # Generate a random distance between 80 km and 120 km
        if random_distance is None:
            was_none = True
            random_distance = random.uniform(80, 110)

    
        # Calculate the coordinates of the random point
        
        point = (reference_camera[0] + random_distance * dir_vector[0], 
                        reference_camera[1] + random_distance * dir_vector[1], 
                        reference_camera[2] + random_distance * dir_vector[2])
        
        if was_none:
            return list(point), random_distance, dir_vector
        else:
            return list(point)
        
    
    
    def generate_direction_vector(self):

        #only z coordination is -1 because object should falling down not upper
        return  [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 0)]


    def create_point_with_direction(self):
        """
        Creates a new point in a random direction from the given 3D point.
        The distance of the new point from the start_point is between 0.01 and 0.5 km.

        :param start_point: A tuple of 3 floats representing the starting 3D point (x, y, z).
        :return: A tuple of 3 floats representing the new 3D point.
        """

        # Convert distance from km to the same unit as the start_point (assuming meters)
        distance = random.uniform(self.range_of_fragment_spread[0], self.range_of_fragment_spread[1] )  # 0.01 - 0.5 km in meters

        # Create a random unit vector for direction
        random_direction = np.random.randn(3)  # random vector
        random_direction /= np.linalg.norm(random_direction)  # normalize to unit vector

        # Calculate new point
        #new_point = np.array(start_point) + distance * random_direction

        #print(distance * random_direction)

        return distance * random_direction



    def generate_cloud_of_fragments(self):

        cloud_of_fragments = []
        print(f"                    starting generating fragments")
        print('--------------------------------------------------------------------------')
        for i in range(0, self.num_of_fragments):
            

            fragment_point = self.create_point_with_direction()
            angle = random.uniform(self.angle[0], self.angle[1]) if isinstance(self.angle, list) else self.angle
            debris_size = random.uniform(self.debris_size[0], self.debris_size[1]) if isinstance(self.debris_size, list) else self.debris_size
            initial_velocity = random.uniform(self.velocity[0], self.velocity[1]) if isinstance(self.velocity, list) else self.velocity
            mass = random.uniform(self.mass[0], self.mass[1]) if isinstance(self.mass, list) else self.mass
            drag_coef = random.uniform(self.drag_coef[0], self.drag_coef[1]) if isinstance(self.drag_coef, list) else self.drag_coef
            surface_area = random.uniform(self.surface_area[0], self.surface_area[1]) if isinstance(self.surface_area, list) else self.surface_area
            lift_coef = random.uniform(self.lift_coef[0], self.lift_coef[1]) if isinstance(self.lift_coef, list) else self.lift_coef


            print(f"Fragment {i + 1}")
            print(f"Angle: {angle}")
            print(f"Debris Size: {debris_size}")
            print(f"Initial Velocity: {initial_velocity}")
            print(f"Mass: {mass}")
            print(f"Drag Coefficient: {drag_coef}")
            print(f"Surface Area: {surface_area}")
            print(f"Lift Coefficient: {lift_coef}")
            print('')

            cloud_of_fragments.append(Fragment(id_=i,location_of_fragment=fragment_point, angle=angle,
                                       debris_size=debris_size, initial_altitude=None,
                                       initial_velocity=initial_velocity, camera_fps=self.amos_time_interval,
                                       focal_length=self.focal_length, pixel_size=self.pixel_size, num_of_frames=self.num_of_frames,
                                       station1_coord_lon_lat=None, station2_coord_ra_dec=None,
                                       start_obst_time=None, mass=mass, drag_coef=drag_coef,
                                       surface_area=surface_area, lift_coef=lift_coef))
            
        return cloud_of_fragments
    
    def altitude_from_geocentric(self, X, Y, Z):
                """
                Calculate the altitude from geocentric coordinates.
                
                :param X: Geocentric X coordinate in kilometers.
                :param Y: Geocentric Y coordinate in kilometers.
                :param Z: Geocentric Z coordinate in kilometers.
                :return: Altitude above Earth's surface in kilometers.
                """
                R_earth = 6371  # Earth's mean radius in kilometers
                r = math.sqrt(X**2 + Y**2 + Z**2)
                altitude = r - R_earth
                return altitude


    def generate_data(self, get_3d=False):
        '''
            Generate synthetic data.

            :param get_3d: Boolean indicating whether to generate 3D data.

            :return: List of generated data.
        '''

        data = []
        alt = []
        down = []

        

        for i in range(0, self.num_sets):
            
            cloud_of_fragments = []
        
            if len(cloud_of_fragments) > 0:
                for fragment in cloud_of_fragments:
                    del fragment

            # creating direction vector of whole cloud of fragments
            dir_vector = self.generate_direction_vector()

            # creating observatory times of cloud of fragments
            obst_times_ = [self.add_mls_to_obs_time(datetime.strptime(self.start_obst_time, "%Y-%m-%d %H:%M:%S.%f"), self.amos_time_interval * i) for i in range(0, self.num_of_frames)]

            camera1_coord = self.get_staton_eloc(self.station1_coord_lon_lat['lon'],self.station1_coord_lon_lat['lat'], self.station1_coord_lon_lat['h'])
            camera2_coord = self.get_staton_eloc(self.station2_coord_lon_lat['lon'],self.station2_coord_lon_lat['lat'], self.station2_coord_lon_lat['h'])

            obst_times = Time(obst_times_, format="datetime")
      
            station1GCRSVectorArr = SkyCoord(camera1_coord.get_gcrs(obstime=obst_times),
                                   frame="gcrs").cartesian / au.m
            
            station2GCRSVectorArr = SkyCoord(camera2_coord.get_gcrs(obstime=obst_times),
                                   frame="gcrs").cartesian / au.m
            
            station1GCRSVectorArr = [[x.x /1000, x.y/1000, x.z/1000] for x in station1GCRSVectorArr]
            station2GCRSVectorArr = [[x.x/1000, x.y/1000, x.z/1000]  for x in station2GCRSVectorArr]
            
            start_point, random_distance, _ = self.generate_point(station1GCRSVectorArr[0], dir_vector=None)

            
            # get sample point with altitude as real debris is cetected
            
            
            while self.altitude_from_geocentric(start_point[0], start_point[1], start_point[2]) < 80 or self.altitude_from_geocentric(start_point[0], start_point[1], start_point[2]) > 120:
                start_point, random_distance, dir_vector = self.generate_point(station1GCRSVectorArr[0])


            cloud_of_fragments = self.generate_cloud_of_fragments()
            
            was_removed = None
            print('\n---------------------------- Generating fragments --------------------------------------\n')
            for i in tqdm(range(0, self.num_of_frames)):
                new_centre_point = self.generate_point(station1GCRSVectorArr[i], random_distance=random_distance, dir_vector=dir_vector)
                
                
                was_removed = None
                for j in range(0, self.num_of_fragments):
                    
                    try:
                        cloud_of_fragments[j].get_fragmet_data(new_centre_point, i, station1GCRSVectorArr[i],station2GCRSVectorArr[i], dir_vector)
                    except ValueError:
                        print("This fragment needed to be removed")
                        was_removed = j
                        pass

                if was_removed is not None:
                    del cloud_of_fragments[j]
                    self.num_of_fragments -= 1

            tmp = {'keypoints0': [], 'scores0':[], 'descriptors0':[[],[],[],[],[],[]],'keypoints1':[], 'scores1':[], 'descriptors1':[[],[],[],[],[],[]], 'fno':[],'id_':[],'carthesian': []}

            for i in range(0, self.num_of_fragments):
                ress = cloud_of_fragments[i].get_final_dict(self.station1_coord_lon_lat, self.station2_coord_lon_lat, obst_times)

                for key in ress.keys():
                    if key == 'descriptors0' or key == 'descriptors1':
                        for j in range(0, len(ress[key])): 

                            tmp[key][j].extend(ress[key][j])
                    else:
                        tmp[key].extend(ress[key])

            data.append(tmp)

            alt.append([fr.alt for fr in cloud_of_fragments])
            down.append([fr.downrange for fr in cloud_of_fragments])

            for fragment in cloud_of_fragments:
                    del fragment

        return data, alt, down

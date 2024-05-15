import os
import numpy as np
import pandas as pd
import pickle
import torch
import yaml
from datetime import datetime, timedelta
from itertools import chain
from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET
from superglue.models.superglue import SuperGlue

validate = {
    "F0": "F0", "F1": "F1", "F2": "F2", "F3": "F3", "F4": "F4",
    "C0": "C3", "C1": "C0", "C2": "C4", "C3": "C10", "C4": "C6",
    "C5": "C9", "B0": "B0", "B1": "C8", "B2": "C5",
    "N0": "C2", "S0": "C7", "T0": "T0"
}

occ = {'true': {}, 'false': {}}

def process_data_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a data frame to prepare for matching by normalizing and aligning columns.

    Args:
        df (pd.DataFrame): Data frame to process.

    Returns:
        pd.DataFrame: Processed data frame.
    """
    # Group by 'obst_time' and aggregate 'alt' and 'Az' into lists
    df = df.groupby('obst_time').agg({
        'alt': list,
        'Az': list,
        'intensity': list,
        'GaussianHeight': list, 
        'GaussianWidth': list,  
        'GaussianBaseline': list,   
        'NumOversaturatedPixels': list
    }).reset_index()

    df.columns = ['obst_time', 'xc', 'yc', 'intensity', 'gh', 'gw', 'gb', 'nop']

    mask = df['xc'].apply(lambda x: len(x) >= 17)

    # Apply the mask to filter the DataFrame
    df = df[mask]

    return df

def print_to_log(message: str, log_file: str):
    """
    Writes a message to a specified log file.
    
    Args:
        message (str): The message to log.
        log_file (str): Path to the log file where the message will be written.
    """
    with open(log_file, 'a') as file:
        file.write(f"{message}\n")

def ensure_folder_exists(folder_path: str):
    """
    Ensures that a folder exists on the filesystem; if not, creates it.
    
    Args:
        folder_path (str): The path to the folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def loop_in_folder(where: str, what: str) -> str:
    """
    Loops through all files in a specified directory, looking for a specified string in YAML files.
    Returns a specific part of the filename where the string is found.

    Args:
        where (str): The directory path to search for files.
        what (str): The string to search for within the YAML files.

    Returns:
        str: Part of the filename where the string was found or None if not found.
    """
    folder_path = where
   
    # Loop through all files in the specified folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)  # Full path to the file
        
        # Check if it's a file (and not a directory)
        text = find_string_in_yaml(file_path, what)
        if text is not None:
            return text

def find_string_in_yaml(filename: str, target_string: str) -> str:
    """
    Searches for a target string within a YAML file and returns a substring of the filename if found.

    Args:
        filename (str): The path to the YAML file.
        target_string (str): The string to search for within the file.

    Returns:
        str: Substring from the filename or None if the string is not found.
    """
    
    with open(filename, 'r') as file:
        data = file.read()
        if target_string in data:
            t = filename.split('_')[-1].split('.')[0]
    
            return t


def display_matches(kpts0: np.ndarray, kpts1: np.ndarray, matches: np.ndarray, conf: np.ndarray, xmls1: str, xmls2: str):
    """
    Displays matching keypoints information, enhances visibility with structured output and logs results.
    
    Args:
        kpts0 (np.ndarray): Array of keypoints from the first image.
        kpts1 (np.ndarray): Array of keypoints from the second image.
        matches (np.ndarray): Array indicating which keypoints from kpts0 match with kpts1.
        conf (np.ndarray): Array of confidence scores for each match.
        xmls1 (str): File path to site 1 data
        xmls2 (str): File path to site 2 data
        
    Returns:
        tuple: A tuple containing occurrences (true and false matches) and count of valid matches.
    """
    valid_matches = matches >= 0
    count = 0
    counter = 0
    
    print("-" * 50)
    print("Pair Analysis Results")
    print("-" * 50)
    
    for i, match in enumerate(valid_matches):
        if match:
            counter += 1
            kp0 = kpts0[i]
            kp1 = kpts1[matches[i]]
            confidence = conf[i]
            
            t1 = loop_in_folder(xmls1, str(kp0[0]))
            t2 = loop_in_folder(xmls2, str(kp1[0]))
            
            if validate.get(t1) == t2:
                print(f"Valid Match {counter}:")
                print(f"Keypoint 1: {kp0}, Keypoint 2: {kp1}, Confidence: {confidence:.2f}")
                count += 1

                if str(t1) + str(t2) in occ['true'].keys():
                    print(t1,t2)
                    occ['true'][str(t1) + str(t2)] += 1
                else:
                    occ['true'][str(t1) + str(t2)] = 1
            else:
                if str(t1) + str(t2) in occ['false'].keys():
                    occ['false'][str(t1) + str(t2)] += 1
                else:
                    occ['false'][str(t1) + str(t2)] = 1

            
    
    # Log to file
    log_results(counter, count)
    print("Summary of occurrences:", occ)
    print(f"Total Valid Matches: {count}, Total Checked: {counter}")
    return occ, count


def log_results(counter, count):
    """
    Logs match results to a text file.
    
    Args:
        counter (int): Total number of matches checked.
        count (int): Total number of valid matches found.
    """
    divider = '*' * 50
    print_to_log(divider, 'out/out.txt')
    print_to_log(f"Occurrences: {occ}", 'out/out.txt')
    print_to_log(f"Valid Matches: {count}, Checked Matches: {counter}", 'out/out.txt')


def match(data_loader: DataLoader, superglue: torch.nn.Module, device: str, xmls1: str, xmls2: str):
    """
    Processes data through a SuperGlue model and logs matching information.
    
    Args:
        data_loader (DataLoader): DataLoader containing the dataset to process.
        superglue (torch.nn.Module): The SuperGlue model to use for processing data.
        device (str): Device ('cuda' or 'cpu') to use for computation.
        xmls1 (str): path to xml file
        xmls2 (str): path to xml file
    
    Returns:
        Tuple: Dictionary of occurrences and the highest number of matches found.
    """
    with torch.no_grad():
        superglue.eval()
        matches_num = []
        for pred in data_loader:
            
            
            processed_pred = {}

            for k, v in pred.items():

                if k not in ['file_name', 'image0', 'image1']:

                    if isinstance(v, torch.Tensor):
                         processed_pred[k] = v.to(device) 
                    else:

                        if k == 'all_matches':
                            concatenated = torch.cat([torch.cat(inner_list, dim=0) for inner_list in v[0]], dim=0)
                            processed_pred[k] = concatenated.to(device)
                     

                        elif k in ['scores0', 'scores1']:
                            processed_pred[k] = torch.tensor(v, dtype=torch.float64).to(device)
                        
                        else:
                            concatenated_list = [torch.cat(tuple(inner_list), dim=0).to(device) for inner_list in v]
                            processed_pred[k] = torch.stack(concatenated_list)


            data = superglue( processed_pred)
            for k, v in  processed_pred.items():
                processed_pred[k] = v
            pred = {**processed_pred, **data}
            
            kpts0, kpts1 = pred['keypoints0'].cpu().numpy(), pred['keypoints1'].cpu().numpy()
            matches, conf = pred['matches0'].cpu().detach().numpy(), pred['matching_scores0'].cpu().detach().numpy()
            occ, counter = display_matches(kpts0, kpts1, matches, conf, xmls1, xmls2)
            matches_num.append(counter)

    return occ, max(matches_num)


def prepare_training_data(df: pd.DataFrame) -> DataLoader:
    """
    Prepares training data from a DataFrame to be used with a SuperGlue model.
    
    Args:
        df (pd.DataFrame): DataFrame containing keypoints and other relevant data.
    
    Returns:
        DataLoader: DataLoader with the prepared data ready for model consumption.
    """
    
    data = []

    sc = np.ones((17, 1), dtype=np.float64)
    kp0 = []
    kp1 = []
    desc1 = []
    desc0 = []

    all_matches = np.concatenate([
        np.array([[x, x] for x in range(17)]),
        np.zeros((17, 1), dtype=np.int64),  #
        np.zeros((17, 1), dtype=np.int64)
    ], axis=1)

    for i, row in df.iterrows():
        
            kp0 = []
            kp1 = []
            desc1 = []
            desc0 = []
            [kp0.append(np.array([x,y])) for x,y  in zip(row['xc_x'], row['yc_x'])]
            
            [kp1.append(np.array([x,y])) for x,y  in zip(row['xc_y'], row['yc_y'])]

            desc0.append([[gh,gw,gb,nop,int] for gh,gw,gb,nop,int  in zip(row['gh_x'], row['gw_x'], row['gb_x'], row['nop_x'], row['intensity_x'])])
            desc1.append([[gh,gw,gb,nop,int] for gh,gw,gb,nop,int  in zip(row['gh_y'], row['gw_y'], row['gb_y'], row['nop_y'], row['intensity_y'])])


            desc0 = np.array(desc0)
            desc1 = np.array(desc1)
       
            desc0 = desc0.reshape(desc0.shape[2], desc0.shape[1])
            desc1 = desc1.reshape(desc1.shape[2], desc1.shape[1])
            
            data.append({'keypoints0': kp0, 'keypoints1': kp1, 'descriptors0': desc0,'descriptors1': desc1, 'scores0':sc, 'scores1': sc, 'all_matches': all_matches})

    return torch.utils.data.DataLoader(dataset=data, shuffle=False, batch_size=1, drop_last=True)

def process_data_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a data frame to prepare for matching by normalizing and aligning columns.

    Args:
        df (pd.DataFrame): Data frame to process.

    Returns:
        pd.DataFrame: Processed data frame.
    """
    # Group by 'obst_time' and aggregate 'alt' and 'Az' into lists
    df = df.groupby('obst_time').agg({
        'alt': list,
        'Az': list,
        'intensity': list,
        'GaussianHeight': list, 
        'GaussianWidth': list,  
        'GaussianBaseline': list,   
        'NumOversaturatedPixels': list
    }).reset_index()

    df.columns = ['obst_time', 'xc', 'yc', 'intensity', 'gh', 'gw', 'gb', 'nop']

    mask = df['xc'].apply(lambda x: len(x) >= 17)

    # Apply the mask to filter the DataFrame
    df = df[mask]

    return df

def get_data_from_folder(folder_path_yaml, folder_path_xml):

    """
    Reads and merges data from YAML and XML files located in specified directories.

    Args:
        folder_path_yaml (str): Path to the directory containing YAML files.
        folder_path_xml (str): Path to the directory containing XML files.

    Returns:
        pd.DataFrame: Merged data frame containing data from both YAML and XML files.
    """

    dfs = []

    

    xmls = os.listdir(folder_path_xml)
    # Loop through all files in the specified folder
    for filename in os.listdir(folder_path_yaml):
        file_path = os.path.join(folder_path_yaml, filename)  # Full path to the file

        print(f'loading file: {file_path}')
        fr = file_path.split('_')[-2]
        file_xml = ''
        for xml in xmls:
            if fr in xml:
                file_xml = xml
                print(xml)
        
        # Check if it's a file (and not a directory)
        df_yaml = (get_yaml_df(file_path))
        df_xml = get_xml_df(folder_path_xml + '/' + file_xml)

        df_yaml['fno'] = df_yaml['fno'].astype(int)
        df_xml['fno'] = df_xml['fno'].astype(int)

        df_xml['Az'] = df_xml['Az'].astype(float)
        df_xml['alt'] = df_xml['alt'].astype(float)
        df_fin = df_yaml.merge(df_xml, on='fno')

        dfs.append(df_fin)

    merged_df = pd.concat(dfs)  # Concatenate your dataframes
    merged_df = merged_df.sort_values(by='obst_time')
    merged_df['Id'] = merged_df.groupby('obst_time').ngroup()

    return merged_df

def get_time_intersection(df1: pd.DataFrame, df2: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Adjusts two data frames to only include data in the overlapping observation times.

    Args:
        df1 (pd.DataFrame): First data frame to intersect.
        df2 (pd.DataFrame): Second data frame to intersect.

    Returns:
        tuple: Tuple containing the intersected data frames.
    """
    
    df1['obst_time'] = pd.to_datetime(df1['obst_time'])
    df2['obst_time'] = pd.to_datetime(df2['obst_time'])

    ranges1 = df1['obst_time'].agg(['min', 'max'])
    ranges2 = df2['obst_time'].agg(['min', 'max'])

    df1 = df1[(df1['obst_time'] >= ranges2['min']) & (df1['obst_time'] <= ranges2['max'])]
    df2 = df2[(df2['obst_time'] >= ranges1['min']) & (df2['obst_time'] <= ranges1['max'])]
    
    
    return df1 , df2

def get_normalised_list(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Normalizes a list of numbers stored within a column of a DataFrame across all rows.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column_name (str): The name of the column containing lists of numbers to normalize.

    Returns:
        pd.DataFrame: DataFrame with the specified column normalized.
    """
    # Extract all values from the lists in the specified column to find the global min and max
    all_values = [item for sublist in df[column_name].tolist() for item in sublist]
    min_val = min(all_values)
    max_val = max(all_values)

    # Normalize the values within each list based on the global min and max
    def normalize(values):
        return [(float(v) - min_val) / (max_val - min_val) if max_val > min_val else 0 for v in values]

    df[column_name] = df[column_name].apply(normalize)
    return df


def get_yaml_df(filename: str) -> pd.DataFrame:
    """
    Parses a YAML file and converts it into a DataFrame.

    Args:
        filename (str): Path to the YAML file.

    Returns:
        pd.DataFrame: DataFrame representing the data from the YAML file.
    """
    
    with open(filename) as yaml_file:
        yaml_content = yaml.load(yaml_file, Loader=yaml.FullLoader)

    event_start_str = yaml_content['EventStartTime'] # The event start time as a string
    event_start_time = datetime.strptime(event_start_str, "%Y-%m-%d %H:%M:%S.%f")  # Convert to datetime object
    df_trail = pd.DataFrame(yaml_content['Trail'] )
    # Calculate the obst_time for each row in the DataFrame
    df_trail['obst_time'] = df_trail['fno'].apply(lambda x: event_start_time + timedelta(seconds=x * 0.05))

    return df_trail


def get_xml_df(filename: str) -> pd.DataFrame:
    """
    Parses an XML file and converts it into a DataFrame.

    Args:
        filename (str): Path to the XML file.

    Returns:
        pd.DataFrame: DataFrame representing the data from the XML file.
    """
    
    # read the file
    with open(filename, 'r') as f:
        xml_data = f.read()

    root = ET.fromstring(xml_data)

    data = []

    # parsing logic
    for ua2_objects in root.findall('ua2_objects'):
        for ua2_object in ua2_objects.findall('ua2_object'):
            for ua2_fdata2 in ua2_object.findall('ua2_objpath/ua2_fdata2'):
                fno = ua2_fdata2.get('fno')
                az = ua2_fdata2.get('az')
                ev = ua2_fdata2.get('ev')
                
                
                # Append the data to the list
                data.append([fno, az, ev ])

    df = pd.DataFrame(data, columns=['fno', 'Az', 'alt'])
    
    return df


def find_matches(first_site_yaml_path: str, first_site_xml_path: str,
                 second_site_yaml_path: str, second_site_xml_path: str):
    """
    Processes and matches data from two different sites using the SuperGlue model.


        first_site_yaml_path (str): Directory path containing YAML files for the first site.
        first_site_xml_path (str): Directory path containing XML files for the first site.
        second_site_yaml_path (str): Directory path containing YAML files for the second site.
        second_site_xml_path (str): Directory path containing XML files for the second site.

    Returns:
        tuple: A tuple containing the dictionary of match occurrences and the highest number of matches found.
    """
    
    # Load data from specified folders
    df_hk = get_data_from_folder(first_site_yaml_path, first_site_xml_path)
    df_mk = get_data_from_folder(second_site_yaml_path, second_site_xml_path)

    # Find the intersection of observation times between two datasets
    df_hk, df_mk = get_time_intersection(df_hk, df_mk)

    # Process the data frames to standardize and prepare for matching
    df_hk = process_data_frame(df_hk)
    df_mk = process_data_frame(df_mk)

    # Merge the processed data frames on observation time for matching
    merged_df = pd.merge(df_hk, df_mk, on='obst_time', how='inner')

    # Normalize specified columns in the merged data frame
    for column in ['gh_x', 'gw_x', 'gb_x', 'nop_x', 'intensity_x', 'gh_y', 'gw_y', 'gb_y', 'nop_y', 'intensity_y']:
        merged_df = get_normalised_list(merged_df, column)

    # Prepare the training data from the merged and normalized data frame
    data = prepare_training_data(merged_df.copy())

    # Determine the computation device based on availability of CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Configuration for the SuperGlue model
    config = {
        'superglue': {
            'sinkhorn_iterations': 200,
            'match_threshold': 0.0001
        }
    }

    # Initialize and load the pretrained SuperGlue model
    superglue = SuperGlue(config.get('superglue', {}))
    superglue.load_state_dict(torch.load('superglue/weights/final_boss.pth', map_location=torch.device('cpu')))
    superglue.double()

    # Use CUDA if available
    if torch.cuda.is_available():
        superglue.cuda()

    # Ensure the output directory exists
    ensure_folder_exists('out')

    # Match data using the SuperGlue model and return the results
    return match(data, superglue, device,first_site_xml_path,second_site_xml_path)

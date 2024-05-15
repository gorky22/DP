import os
import numpy as np
import pandas as pd
import argparse
import yaml
import plotly.express as px
from position_matcher import meteor
from typing import List, Dict, Optional
import match
#python3 main.py yamls_/HK xmls/HK yamls_/MK xmls/MK
def get_pair_codes(trajectories: Dict[str, Dict[str, int]], type: str = 'true') -> List[str]:
    """
    Extracts pairs of trajectories that meet a certain threshold of occurrences.

    Args:
        trajectories (Dict[str, Dict[str, int]]): Dictionary of trajectory data.
        type (str): Type of trajectory, either 'true' or 'false'.

    Returns:
        List[str]: List of fragment codes that meet the specified criteria.
    """
    return [pair for pair, count in trajectories[type].items() if count >= 10]

def find_file_by_fragment_code(folder_path: str, fragment_code: str) -> Optional[str]:
    """
    Searches for a specific file in a folder based on a fragment code.

    Args:
        folder_path (str): The directory path to search for files.
        fragment_code (str): The fragment code to match in file names.

    Returns:
        Optional[str]: Path to the file if found, None otherwise.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):  # Filter for XML files
            fr = filename.split('.')[0].split('_')[-1]
            if fragment_code == fr:
                return os.path.join(folder_path, filename)
    return None

def get_station_coords(yaml_file_path: str) -> Optional[List[float]]:
    """
    Reads GPS coordinates from a YAML file.

    Args:
        yaml_file_path (str): Path to the YAML file.

    Returns:
        Optional[List[float]]: List containing longitude, latitude, and altitude.
    """
    try:
        with open(yaml_file_path, 'r') as file:
            yaml_content = yaml.safe_load(file)
            return [yaml_content['Longitude'], yaml_content['Latitude'], yaml_content['Altitude']]
    except Exception as e:
        print(f"Error reading {yaml_file_path}: {e}")
        return None

def save_all_data_to_single_csv(data: Dict[str, np.ndarray], filename: str = 'out/all_data.csv') -> None:
    """
    Consolidates all data arrays into a single CSV file.

    Args:
        data (Dict[str, np.ndarray]): Dictionary of data arrays.
        filename (str): Path to save the CSV file.

    Returns:
        None
    """
    all_data = []
    for key, array in data.items():
        df = pd.DataFrame(array, columns=['X', 'Y', 'Z'])
        df['Fragment Code'] = key
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv(filename, index=False)
    print(f"All data saved to {filename}.")

def get_final_fragments_data(fragments: List[str], first_site_yaml: str, first_site_xmls: str, second_site_yaml: str, second_site_xmls: str) -> Dict[str, np.ndarray]:
    """
    Processes and retrieves final fragments data for given sites.

    Args:
        fragments (List[str]): List of fragment codes.
        first_site_yaml (str): Directory path for first site YAML files.
        second_site_xmls (str): Directory path for second site XML files.
        second_site_yaml (str): Directory path for second site YAML files.
        second_site_xmls (str): Directory path for second site XML files.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing data arrays for each fragment.
    """
    dataset = {}
    station1 = get_station_coords(os.path.join(first_site_yaml, os.listdir(first_site_yaml)[0]))
    station2 = get_station_coords(os.path.join(second_site_yaml, os.listdir(second_site_yaml)[0]))

    for fragment in fragments:
        path1 = find_file_by_fragment_code(first_site_xmls, fragment[:2])
        path2 = find_file_by_fragment_code(second_site_xmls, fragment[-2:])

        if path1 is None or path2 is None:
            continue

        meteor_instance = meteor.Meteor(path1, path2, station1, station2, "M20191015_073816")
        data = np.array(meteor_instance.df_with_trajectory.real_coords.to_list())
        dataset[fragment] = data

    return dataset

class MultiTrajectoryPlotter:
    def __init__(self, trajectories: Dict[str, List[np.ndarray]], type: str = 'true') -> None:
        """
        Initializes the MultiTrajectoryPlotter with a set of trajectories and their type.

        Parameters:
            trajectories (Dict[str, List[np.ndarray]]): Dictionary mapping fragment codes to lists of coordinates.
            type (str): Type of the trajectories, either 'true' for correctly matched or 'false' for incorrectly matched.
        """
        self.trajectories = trajectories
        self.type = type

    def plot_trajectories(self) -> None:
        """
        Generates and displays a 3D scatter plot of the trajectories stored in the object.

        Uses Plotly to create the visualization, differentiating trajectories by color.
        """
        # Prepare DataFrame for plotting
        data = [{'X': coord[0], 'Y': coord[1], 'Z': coord[2], 'Fragment code': name}
                for name, coords in self.trajectories.items() for coord in coords if not np.isinf(coord).any()]
        df = pd.DataFrame(data)

        # Create the 3D scatter plot
        fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='Fragment code',
                            title=f'Trajectory of {self.type}-matched Fragments in Time',
                            labels={'X': 'X [m]', 'Y': 'Y [m]', 'Z': 'Z [m]'},
                            width=1000, height=600)
        fig.update_traces(marker=dict(size=3, line=dict(width=0.5)))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), scene=dict(
            xaxis_title='X [m]', yaxis_title='Y [m]', zaxis_title='Z [m]'))
        fig.show()
        fig.write_html(f'out/fragments_{self.type}.html')


def main(first_site_yaml: str, first_site_xmls: str, second_site_yaml: str, second_site_xmls: str) -> None:
    """
    Main function to process and plot trajectories from YAML and XML files for two sites.

    Parameters:
        first_site_yaml (str): Directory path for first site YAML files.
        first_site_xmls (str): Directory path for first site XML files.
        second_site_yaml (str): Directory path for second site YAML files.
        second_site_xmls (str): Directory path for second site XML files.
    """
    trajectories, best_result = match.find_matches(first_site_yaml, first_site_xmls, second_site_yaml, second_site_xmls)
    fragments_f = get_pair_codes(trajectories, type='false')
    fragments_t = get_pair_codes(trajectories, type='true')
    datas_t = get_final_fragments_data(fragments_t, first_site_yaml, first_site_xmls, second_site_yaml, second_site_xmls)
    datas_f = get_final_fragments_data(fragments_f, first_site_yaml, first_site_xmls, second_site_yaml, second_site_xmls)

    save_all_data_to_single_csv(datas_t, 'out/final.csv')
    plotter = MultiTrajectoryPlotter(datas_t)
    plotter.plot_trajectories()
    plotter2 = MultiTrajectoryPlotter(datas_f, type='false')
    plotter2.plot_trajectories()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and plot trajectories from YAML and XML files for two sites.")
    parser.add_argument('first_site_yaml', help='Directory path for first site YAML files')
    parser.add_argument('first_site_xmls', help='Directory path for first site XML files')
    parser.add_argument('second_site_yaml', help='Directory path for second site YAML files')
    parser.add_argument('second_site_xmls', help='Directory path for second site XML files')
    args = parser.parse_args()
    main(args.first_site_yaml, args.first_site_xmls, args.second_site_yaml, args.second_site_xmls)


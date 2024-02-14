import numpy as np
import torch
import pickle
from superglue.models import superglue_eval as superglue


def load_data(file_path):
    """
    Load data from a pickle file.
    
    Args:
        file_path (str): Path to the pickle file.
    
    Returns:
        object: Loaded data.
    """
    with open(file_path, 'rb') as fp:
        return pickle.load(fp)


def save_data(data, file_path):
    """
    Save data to a pickle file.
    
    Args:
        data (object): Data to be saved.
        file_path (str): Path to save the pickle file.
    """
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def normalize_array(arr):
    """
    Normalize an array to have values between 0 and 1.
    
    Args:
        arr (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Normalized array.
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val)


def tensorify_pairs(pairs):
    """
    Converts pairs of data to tensors.

    Args:
        pairs (dict): Dictionary of data pairs.

    Returns:
        dict: Dictionary with tensor data.
    """
    for k in pairs:
        if isinstance(pairs[k], (list, tuple)):
            pairs[k] = [torch.tensor(item) for item in pairs[k]]
        pairs[k] = torch.stack(pairs[k])
    return pairs


def perform_matching(sg, pairs):
    """
    Perform matching using the SuperGlue model.

    Args:
        sg: SuperGlue model.
        pairs (dict): Dictionary of data pairs.

    Returns:
        tuple: Tuple of keypoints, matches, and confidence scores.
    """
    pred = sg(pairs)
    pred = {**pred, **pairs}
    kpts0, kpts1 = pred['keypoints0'].cpu().numpy(), pred['keypoints1'].cpu().numpy()
    matches, conf = pred['matches0'].cpu().detach().numpy(), pred['matching_scores0'].cpu().detach().numpy()
    return kpts0, kpts1, matches, conf


def display_matches(kpts0, kpts1, matches, conf):
    """
    Display matching keypoints information.

    Args:
        kpts0, kpts1 (numpy.ndarray): Arrays of keypoints.
        matches (numpy.ndarray): Array of matches.
        conf (numpy.ndarray): Array of confidence scores.
    """
    x = matches >= 0
    counter = 0
    for i, match in enumerate(x):
        if match:
            counter += 1
            kp0 = kpts0[i]
            kp1 = kpts1[matches[i]]
            confidence = conf[i]
            print(f"Match {i + 1}:")
            print(f"Keypoint 1: {kp0}")
            print(f"Keypoint 2: {kp1}")
            print("-" * 20)


if __name__ == "__main__":
    """
    Main function to run the process.
    """

    torch.set_grad_enabled(False)

    # Load data
    pairs = load_data('data/test_data.pickle')
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device "{}"'.format(device))

    # Configuration for SuperPoint and SuperGlue
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        }
    }

    print('test data: ')
    
    for i in range(0, len(pairs['keypoints0'])):
        print(pairs['keypoints0'][i], pairs['keypoints1'][i])

    
    # Convert pairs to tensors
    pairs = tensorify_pairs(pairs)
    
    # Initialize SuperGlue model
    sg = superglue.SuperGlue(config.get('superglue', {}))

    # Perform matching
    kpts0, kpts1, matches, conf = perform_matching(sg, pairs)

    # Display matches
    display_matches(kpts0, kpts1, matches, conf)



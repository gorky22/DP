import torch
from torch.autograd import Variable
import torch.utils.data
import numpy as np
import pickle
import models.superglue as Superglue
from models.utils import AverageTimer


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

def print_and_log(*args, **kwargs):

    """
    print data to terminal and also to log file

   
    """
    with open('out.txt', 'a') as f:  # Open the file in append mode
        print(*args, **kwargs)  # Print to stdout
        print(*args, **kwargs, file=f)  # Write the same to the file


def prepare_training_data(data):
    """
    Prepare the training data.

    Args:
        data (list): List of data items.

    Returns:
        list: List of processed training data.
    """
    train_data = []
    sc = [1 for _ in range(15)]
    all_matches = np.concatenate([
        np.array([[x, x] for x in range(15)]),
        np.zeros((15, 1), dtype=np.int64),  # Adding empty columns
        np.zeros((15, 1), dtype=np.int64)
    ], axis=1)

    for i in range(len(data)):
        data[i]['keypoints0'] = [tuple(row) for row  in np.array(data[i]['keypoints0'])]
        data[i]['keypoints1'] = [tuple(row) for row in  np.array(data[i]['keypoints1'])]

        descs0 = np.array([normalize_array(x) for x in np.array(data[i]['descriptors0'][:-1])])
        data[i]['descriptors0'] = descs0.T.reshape(15,1500,5)
        descs1 = np.array([normalize_array(x) for x in np.array(data[i]['descriptors1'][:-1])])
        data[i]['descriptors1'] = descs1.T.reshape(15,1500,5)

        for j in range(0,1500):
            kp0 = []
            kp1 = []
            desc0 = []
            desc1 = []

            for k in range(0,15):

                kp0.append(data[i]['keypoints0'][k][j])
                kp1.append(data[i]['keypoints1'][k][j])

                desc0.append(data[i]['descriptors0'][k][j])
                desc1.append(data[i]['descriptors1'][k][j])
            
            
            train_data.append({'keypoints0': kp0, 'keypoints1': kp1, 'descriptors0': desc0,'descriptors1': desc1, 'scores0':sc, 'scores1': sc, 'all_matches': all_matches})



    train_loader = torch.utils.data.DataLoader(dataset=train_data, shuffle=False, batch_size=1, drop_last=True)

    return train_loader


def train_model(train_loader, superglue, optimizer, num_epochs=20):
    """
    Train the SuperGlue model.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        superglue (Module): SuperGlue model.
        optimizer (Optimizer): Optimizer for the model.
        num_epochs (int): Number of epochs for training.
    """
    mean_loss = []
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0
        superglue.double().train()
        epoch_loss = 0
        superglue.double().train()

        for i, pred in enumerate(train_loader):
    
            for k in pred:
                if k != 'file_name' and k!='image0' and k!='image1':
                    if type(pred[k]) == torch.Tensor:
                
                        pred[k] = Variable(pred[k])
                    else:
                    
                        if k != 'scores0' and k != 'scores1':
                            if k == 'all_matches':
                                pred[k] =  torch.cat([torch.cat(inner_list, dim=0) for inner_list in pred[k][0]],[],[])
                            else:
                                pred[k] =  [torch.cat(tuple(inner_list), dim=0) for inner_list in pred[k]]
                        
                        pred[k] = Variable(torch.stack(pred[k]))
                
            data = superglue(pred)
            for k, v in pred.items():
                pred[k] = v
            pred = {**pred, **data}

            if pred['skip_train'] == True: # image has n o keypoint
                continue
            
            # process loss
            
            Loss = pred['loss']
            Loss = Variable(Loss, requires_grad = True)
            epoch_loss += Loss.item()
            mean_loss.append(Loss)

            superglue.zero_grad()
            Loss.backward()
            optimizer.step()

            # for every 50 images, print_and_log progress and visualize the matches
            if (i+1) % 50 == 0:
                print_and_log('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch, 20, i+1, len(train_loader), torch.mean(torch.stack(mean_loss)).item())) 
                mean_loss = []


                kpts0, kpts1 = pred['keypoints0'].cpu().numpy(), pred['keypoints1'].cpu().numpy()
                matches, conf = pred['matches0'].cpu().detach().numpy(), pred['matching_scores0'].cpu().detach().numpy()


                x = matches >= 0
        
                for i, match in enumerate(x):
                
                    if match:
                        kp0 = kpts0[i]
                        kp1 = kpts1[matches[i]]
                        confidence = conf[i]
                        print_and_log(f"Match {i + 1}:")
                        print_and_log(f"Keypoint 1: {kp0}")
                        print_and_log(f"Keypoint 2: {kp1}")
                        print_and_log(f"Confidence: {confidence}")
                        print_and_log("-" * 20)

            # process checkpoint for every 5e3 images
            if (i+1) % 5e3 == 0:
                model_out_path = "model_epoch_{}.pth".format(epoch)
                torch.save(superglue, model_out_path)
                print_and_log ('Epoch [{}/{}], Step [{}/{}], Checkpoint saved to {}' 
                    .format(epoch, 20, i+1, len(train_loader), model_out_path)) 

        # save checkpoint when an epoch finishes
        epoch_loss /= len(train_loader)
        model_out_path = "model_epoch_{}.pth".format(epoch)
        torch.save(superglue, model_out_path)
        print_and_log("Epoch [{}/{}] done. Epoch Loss {}. Checkpoint saved to {}"
            .format(epoch, 20, epoch_loss, model_out_path))


def main():
    """
    Main function to run the process.
    """
    torch.set_grad_enabled(False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print_and_log('Running inference on device "{}"'.format(device))

    # Configuration and initialization
    config = {
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    superglue = Superglue.SuperGlue(config.get('superglue', {}))
    if torch.cuda.is_available():
        superglue.cuda()

    # Load and prepare data
    data = load_data('../data/data_1.pkl')
    train_loader= prepare_training_data(data)
    
    # Train model
    optimizer = torch.optim.Adam(superglue.parameters(), lr=0.001)
    train_model(train_loader, superglue, optimizer)


if __name__ == "__main__":
    main()

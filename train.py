import torch
from torch.autograd import Variable
import torch.utils.data
import numpy as np
import pickle
import wandb
from superglue.models.superglue import SuperGlue

def load_data(file_path: str) -> object:
    """
    Load data from a pickle file.
    :param file_path: Path to the pickle file.
    :return: Loaded data.
    """
    with open(file_path, 'rb') as fp:
        return pickle.load(fp)

def normalize_array(arr: np.ndarray) -> np.ndarray:
    """
    Normalize an array to have values between 0 and 1.
    :param arr: Input array.
    :return: Normalized array.
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val)

def print_and_log(*args, **kwargs) -> None:
    """
    Print data to terminal and also log to file.
    """
    print(*args, **kwargs)

def prepare_training_data(data: list) -> tuple:
    """
    Prepare the training data.
    :param data: List of data items.
    :return: Tuple of processed training data.
    """
    train_data = []
    #sc = [float(1) for _ in range(15)]
    sc = np.ones((15, 1), dtype=np.float64)
    all_matches = np.concatenate([
        np.array([[x, x] for x in range(15)]),
        np.zeros((15, 1), dtype=np.int64),  # Adding empty columns
        np.zeros((15, 1), dtype=np.int64)
    ], axis=1)

    for i in range(len(data)):
        data[i]['keypoints0'] = [tuple(row) for row  in np.array(data[i]['keypoints0'])]
        data[i]['keypoints1'] = [tuple(row) for row in  np.array(data[i]['keypoints1'])]


        descs0 = np.array([normalize_array(x) for x in np.array(data[i]['descriptors0'][:-1])])

        data[i]['descriptors0'] = descs0.T.reshape(15,2000,5)
        descs1 = np.array([normalize_array(x) for x in np.array(data[i]['descriptors1'][:-1])])
        data[i]['descriptors1'] = descs1.T.reshape(15,2000,5)

        for j in range(0,2000):
            kp0 = []
            kp1 = []
            desc0 = []
            desc1 = []

            for k in range(0,15):

                kp0.append(data[i]['keypoints0'][k][j])
                kp1.append(data[i]['keypoints1'][k][j])

                desc0.append(data[i]['descriptors0'][k][j])
                desc1.append(data[i]['descriptors1'][k][j])

            desc0 = np.array(desc0)
            desc1 = np.array(desc1)
            desc0 = desc0.reshape(desc0.shape[1], desc0.shape[0])
            desc1 = desc1.reshape(desc1.shape[1], desc1.shape[0])
            train_data.append({'keypoints0': kp0, 'keypoints1': kp1, 'descriptors0': desc0,'descriptors1': desc1, 'scores0':sc, 'scores1': sc, 'all_matches': all_matches})


    train_data = train_data[:int(len(train_data) * 0.1)]
    separation = int(len(train_data) * 0.9)

    train = train_data[:separation]
    test = train_data[separation:]

    train_loader = torch.utils.data.DataLoader(dataset=train, shuffle=True, batch_size=1, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test, shuffle=False, batch_size=1, drop_last=True)

    return train_loader, test_loader


def train_model(train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader,
                superglue: SuperGlue, optimizer: torch.optim.Optimizer, num_epochs: int = 100) -> None:
    """
    Train the SuperGlue model.
    :param train_loader: DataLoader for training data.
    :param test_loader: DataLoader for testing data.
    :param superglue: SuperGlue model.
    :param optimizer: Optimizer for the model.
    :param num_epochs: Number of epochs for training.
    """

    wandb.login(key='cdcdb4205bb5e10f1f0fa08d5079913d8f04fcd7')

    run = wandb.init(
          project="DP_test_same_smaller_lr",
            name=f"experiment_new_working",
          config={
         "learning_rate": 0.0001,
         "architecture": "GNN",
          "dataset": "colab3.pkl",
          "epochs": 40,
          })


    mean_loss = []
    for epoch in range(1, num_epochs + 1):
      epoch_loss = 0
      train_acc = []
      superglue.double().train()  # Ensure the model is in training mode

      for i, pred in enumerate(train_loader):
          processed_pred = {}
          for k, v in pred.items():
              if k not in ['file_name', 'image0', 'image1']:

                  if isinstance(v, torch.Tensor):
                      processed_pred[k] = v.to(device)  # Ensure tensors are on the correct device
                      
                  else:
                      if k == 'all_matches':
                          concatenated = torch.cat([torch.cat(inner_list, dim=0) for inner_list in v[0]], dim=0)
                          processed_pred[k] = concatenated.to(device)

                      elif k in ['scores0', 'scores1']:
                        processed_pred[k] = torch.tensor(v, dtype=torch.float64).to(device)
                   
                      else:
                          concatenated_list = [torch.cat(tuple(inner_list), dim=0).to(device) for inner_list in v]
                          processed_pred[k] = torch.stack(concatenated_list)
                          

          data = superglue(processed_pred)  # Forward pass

          # Compute the loss
          Loss = data['loss']
          #Loss.requires_grad = True
          epoch_loss += Loss.item()
          train_acc.append(data['acc'])
          # Backward pass and optimization
          optimizer.zero_grad()  # Clear existing gradients
          Loss.backward()  # Compute gradients
          optimizer.step()  # Update model parameters

          if (i + 1) % 100 == 0:
              print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, acc: {:.4f}'.format(epoch, num_epochs, i + 1, len(train_loader), Loss.item(),data['acc']))
                
      epoch_loss /= len(train_loader)
      
      acc = []
      with torch.no_grad():
        superglue.eval()
        for i, pred in enumerate(test_loader):

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
            acc.append(data['acc'])


      acc2 = np.average(np.array(acc))
      print_and_log("Epoch [{}/{}] done. Epoch Loss {}. Acc {}. train Acc {} Checkpoint saved to"
          .format(epoch, 20, epoch_loss, acc2, np.average(np.array(train_acc))))
      wandb.log({"loss": epoch_loss, 'acc': acc2, 'train_acc': np.average(np.array(train_acc)) })
    wandb.finish()

config = {'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 80,
        'match_threshold': 0.2
    }}
    
# Setup and initialization for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
superglue = SuperGlue(config.get('superglue', {}))
superglue.to(device)  # Move model to the appropriate device

# Load and prepare data
data = load_data('colab_200.pkl')
train_loader, test_loader = prepare_training_data(data)

# Initialize optimizer
optimizer = torch.optim.Adam(superglue.parameters(), lr=0.001)

# Train the model
train_model(train_loader, test_loader, superglue, optimizer)


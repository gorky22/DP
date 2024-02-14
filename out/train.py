#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
from torch.autograd import Variable

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)
import models.superglue as Superglue
#from test_data_loader import SparseDataset
import torch
import random
#from test_data_loader2 import DataGenerator

import pickle
import numpy as np

#info_pairs = '/Users/damiangorcak/Desktop/dp/dp-satelites/test_graph_matching/SuperGluePretrainedNetwork-master/test.txt'
#input_dir = '/Users/damiangorcak/Desktop/dp/dp-satelites/test_graph_matching/SuperGluePretrainedNetwork-master/test'

#with open(info_pairs, 'r') as f:
#    pairs = [l.split() for l in f.readlines()]

import numpy as np

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
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr



device = 'cuda' if torch.cuda.is_available() and not None else 'cpu'
print('Running inference on device \"{}\"'.format(device))

config = {
            
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
}

#matching = Matching(config).eval().to(device)


timer = AverageTimer(newline=True)

'''
def norm(val,min,max):
    return (val-min)/(max-min)



data = []
min = 0
max = 1300
for i in range(0,100):
    for j in range(0,30):
        
      
        #keypoints0 = [[980.7389, 618.5122,], [954.4825, 84.2555],  [1054.4825, 834.2555]]
        #descriptors0 = [[4.112600e+04, 3.078595e+02, 3.785950e+01],[117.7602,  21.4774, 333.4774],[ 1.3902,  5.2358, 52.2358],[24.2219, 20.6575, 21.6575],[0., 2., 3.]]
        #scores0 = [1,1,1]

        #keypoints1 = [[591.1545, 190.1858], [54.4825, 184.2555],  [654.4825, 334.2555]]
        #descriptors1 = [[1311.    ,  607.8595,  237.8595],[119.439 ,  21.4774, 456.4774],[ 1.69  ,  5.2358, 52.2358],[  23.2598,  520.6575, 1221.6575],[0., 2., 1.]]
        #scores1 = [1,1,1]

        
        keypoints0 = [[norm(980.7389, min, max), norm(618.5122, min, max)], [norm(954.4825, min, max), norm(84.2555, min, max)], [norm(1054.4825, min, max), norm(834.2555, min, max)]]
        descriptors0 = [[norm(4.112600e+04, min, max), norm(3.078595e+02, min, max), norm(3.785950e+01, min, max)],
                [norm(117.7602, min, max), norm(21.4774, min, max), norm(333.4774, min, max)],
                [norm(1.3902, min, max), norm(5.2358, min, max), norm(52.2358, min, max)],
                [norm(24.2219, min, max), norm(20.6575, min, max), norm(21.6575, min, max)],
                [norm(0., min, max), norm(2., min, max), norm(3., min, max)]]
        scores0 = [1, 1, 1]

        keypoints1 = [[norm(591.1545, min, max), norm(190.1858, min, max)], [norm(54.4825, min, max), norm(184.2555, min, max)], [norm(654.4825, min, max), norm(334.2555, min, max)]]
        descriptors1 = [[norm(1311., min, max), norm(607.8595, min, max), norm(237.8595, min, max)],
                        [norm(119.439, min, max), norm(21.4774, min, max), norm(456.4774, min, max)],
                        [norm(1.69, min, max), norm(5.2358, min, max), norm(52.2358, min, max)],
                        [norm(23.2598, min, max), norm(520.6575, min, max), norm(1221.6575, min, max)],
                        [norm(0., min, max), norm(2., min, max), norm(1., min, max)]]
        scores1 = [1, 1, 1]
        
        keypoints0.append([norm(float(random.randint(min, max)),min,max),norm(float(random.randint(min, max)),min,max)])
        descriptors0[0].append(norm(float(random.randint(min, max)),min,max))
        descriptors0[1].append(norm(float(random.randint(min, max)),min,max))
        descriptors0[2].append(norm(float(random.randint(min, max)),min,max))
        descriptors0[3].append(norm(float(random.randint(min, max)),min,max))
        descriptors0[4].append(norm(float(random.randint(min, max)),min,max))

        scores0.append(1)

        keypoints1.append([norm(float(random.randint(min, max)),min,max),norm(float(random.randint(min, max)),min,max)])
        descriptors1[0].append(norm(float(random.randint(min, max)),min,max))
        descriptors1[1].append(norm(float(random.randint(min, max)),min,max))
        descriptors1[2].append(norm(float(random.randint(min, max)),min,max))
        descriptors1[3].append(norm(float(random.randint(min, max)),min,max))
        descriptors1[4].append(norm(float(random.randint(min, max)),min,max))

        scores1.append(1)
        


        #keypoints0 = torch.tensor(keypoints0)
        #descriptors0 = torch.tensor(descriptors0)
        #scores0 = torch.tensor(scores0)
        ##keypoints1 = torch.tensor(keypoints1)
        #descriptors1 = torch.tensor(descriptors1)
        #scores1 =   torch.tensor(scores1)

    #data.append({'keypoints0': [keypoints0], 'scores0':[scores0], 'descriptors0':[descriptors0],'keypoints1': [keypoints1], 'scores1':[scores1], 'descriptors1':[descriptors1]})
    data.append({'keypoints0': keypoints0, 'scores0':scores0, 'descriptors0':descriptors0,'keypoints1': keypoints1, 'scores1':scores1, 'descriptors1':descriptors1})
'''


#dataGenerator = DataGenerator(1000)

#train_data = dataGenerator.generate_data()



#with open('../data/data.pkl', 'rb') as fp:
#        data = pickle.load(fp)



# for i in range(0,len(train_set)):
#     train_set[i]['keypoints0'] = [tuple(row) for row  in np.array(train_set[i]['keypoints0'])]
#     train_set[i]['keypoints1'] = [tuple(row) for row in  np.array(train_set[i]['keypoints1']]]

#     train_set[i]['descriptors0'] = train_set[i]['descriptors0'][:-1]
#     train_set[i]['descriptors1'] = train_set[i]['descriptors1'][:-1]

#     train_set[i].pop('id_')
#     train_set[i].pop('carthesian')
#     train_set[i].pop('fno')

#     tmp = np.array([[x,x] for x in range(0, len(train_set[i]['keypoints1']))])
#     train_set[i]['all_matches'] = np.concatenate([
#     tmp,
#     np.zeros((22500,1), dtype=np.int64),  # This is effectively adding nothing
#     np.zeros((22500,1), dtype=np.int64)   # This is effectively adding nothing
# ], axis=1)
    
#     train_set[i]['scores0'] = [i for i in range(0,22500)]
#     train_set[i]['scores1'] = [i for i in range(0,22500)]
        
import numpy as np

train_data = []

sc = [1 for i in range(0,15)]
with open('dp-satelites/final_pipe/data/data_1.pkl', 'rb') as fp:
        data = pickle.load(fp)

tmp = np.array([[x,x] for x in range(0, 15)])
all_matches = np.concatenate([
tmp,
np.zeros((15,1), dtype=np.int64),  # This is effectively adding nothing
np.zeros((15,1), dtype=np.int64)   # This is effectively adding nothing
], axis=1)
    
for i in range(0,len(data)):
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

superglue = Superglue.SuperGlue(config.get('superglue', {}))

if torch.cuda.is_available():
    superglue.cuda() # make sure it trains on GPU
else:
    print("### CUDA not available ###")
optimizer = torch.optim.Adam(superglue.parameters(), lr=0.001)
mean_loss = []



# start training
for epoch in range(1, 20+1):
    epoch_loss = 0
    superglue.double().train()
    for i, pred in enumerate(train_loader):
 
        for k in pred:
            if k != 'file_name' and k!='image0' and k!='image1':
                if type(pred[k]) == torch.Tensor:
                    #pred[k] = Variable(pred[k].cuda())

                    pred[k] = Variable(pred[k])
                else:
                    #pred[k] = Variable(torch.stack(pred[k]).cuda())
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

        # for every 50 images, print progress and visualize the matches
        if (i+1) % 50 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch, 20, i+1, len(train_loader), torch.mean(torch.stack(mean_loss)).item())) 
            mean_loss = []

            ### eval ###
            # Visualize the matches.
            #superglue.eval()
            


                        #### try this to change with our features
            kpts0, kpts1 = pred['keypoints0'].cpu().numpy(), pred['keypoints1'].cpu().numpy()
            matches, conf = pred['matches0'].cpu().detach().numpy(), pred['matching_scores0'].cpu().detach().numpy()

            #valid = matches > -1
            #mkpts0 = kpts0[valid]
            #mkpts1 = kpts1[matches[valid]]
            #mconf = conf[valid]

            x = matches >= 0
       
            for i, match in enumerate(x):
              
                if match:
                    kp0 = kpts0[i]
                    kp1 = kpts1[matches[i]]
                    confidence = conf[i]
                    print(f"Match {i + 1}:")
                    print(f"Keypoint 1: {kp0}")
                    print(f"Keypoint 2: {kp1}")
                    print(f"Confidence: {confidence}")
                    print("-" * 20)

        # process checkpoint for every 5e3 images
        if (i+1) % 5e3 == 0:
            model_out_path = "model_epoch_{}.pth".format(epoch)
            torch.save(superglue, model_out_path)
            print ('Epoch [{}/{}], Step [{}/{}], Checkpoint saved to {}' 
                .format(epoch, 20, i+1, len(train_loader), model_out_path)) 

    # save checkpoint when an epoch finishes
    epoch_loss /= len(train_loader)
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(superglue, model_out_path)
    print("Epoch [{}/{}] done. Epoch Loss {}. Checkpoint saved to {}"
        .format(epoch, 20, epoch_loss, model_out_path))
    



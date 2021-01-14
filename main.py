import os,argparse,time
import numpy as np
#from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
import random
import torch.nn as nn

from networks import network 
from networks import Classifier
import utils
from sklearn.preprocessing import normalize



class args():
  pass
args.n_x = 2048
args.n_y = 312 #64 for aPy #85 for AWA #102 for SUN #312 for CUB
args.hid = 500
args.n_z = 50
args.train_batch_size = 59
args.test_batch_size = 59
args.num_tasks = 20 #4 for aPy #5 for AWA #15 for SUN #20 for CUB
args.num_classes = 200 #32 for aPy #50 for AWA #708 for SUN #200 for CUB
args.num_epochs = 100
args.diff = 'yes'
args.lam = 1
args.s_steps = 1
args.d_steps = 1
args.adv = 0.5
args.orth = 0.1
args.ta = 0.01
args.class_epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device
args.checkpoint = './checkpoints/'
args.checkpoint_class = './class_checkpoints/'

args.e_mo = 0.01
args.e_dr = 0.01
args.e_class = 1e-4
args.mom = 0.9

path = './class_checkpoints/class.pth'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
nSamples = 200
replay_nSamples = 50
print('device', device)

from a_vae import A_VAE
from classi import classifier_train

print('Good Boy')
#------------------------------------- data preprocessing ---------------------------------------------#
def dataprocess(data_path):
  with open(data_path, 'rb') as fopen:
     #contents = np.load(fopen, allow_pickle=True, encoding='bytes')
    contents = np.load(fopen, allow_pickle=True, encoding='latin1')
    return contents
#/home/airl-gpu3/chandan_sethu/CVPR_2021_CGZSL/SUN

trainData1 = dataprocess('https://drive.google.com/file/d/1jpcOirBg8675GtuD8Oa4QVczgZ7suDPR/view?usp=sharing')
trainLabels1 = dataprocess('https://drive.google.com/file/d/1Fch_H6FI4Y-6KU7gz8NjXvfOrvc1SvGY/view?usp=sharing')
trainLabelsVectors1 = dataprocess('https://drive.google.com/file/d/1wytjK-9aKreSeEcQqcYAGFJ7ajlK_e41/view?usp=sharing')
testData1 = dataprocess('https://drive.google.com/file/d/1WbsNUgEeiGJY0p2Q4uIK7xKVKqLG0Ooy/view?usp=sharing')
testLabels1 = dataprocess('https://drive.google.com/file/d/1lD6F90hGikP6ROwboImpL91YaVA7tNZm/view?usp=sharing')
ATTR = dataprocess('https://drive.google.com/file/d/1rnoVbA_Hmgcm0c6q5aiiVhxhnSVYgE_L/view?usp=sharing')

'''
trainData1 = dataprocess('/content/drive/My Drive/ColabNotebooks/Replace GAN/Send_to_Shubhankar/trainData')
trainLabels1 = dataprocess('/content/drive/My Drive/ColabNotebooks/Replace GAN/Send_to_Shubhankar/trainLabels')
trainLabelsVectors1 = dataprocess('/content/drive/My Drive/ColabNotebooks/Replace GAN/Send_to_Shubhankar/trainAttributes')
ATTR = dataprocess('/content/drive/My Drive/ColabNotebooks/Replace GAN/Send_to_Shubhankar/dataAttributes')
testData1 = dataprocess('/content/drive/My Drive/ColabNotebooks/Replace GAN/Send_to_Shubhankar/testData')
testLabels1 = dataprocess('/content/drive/My Drive/ColabNotebooks/Replace GAN/Send_to_Shubhankar/testLabels')
'''
#from a_vae import A_VAE
#from classi import classifier_train
print(trainLabels1, 'a', testLabels1)
net = network.net(args).to(device)
#CLASSI = Classifier.CLASSIFIER(args).to(device)

#print(testData1[0], '0')
#torch.save(CLASSI.state_dict(), path)

appr = A_VAE(net, args)
#class_appr = classifier_train(CLASSI, args)


seen_acc = []
unseen_acc = []
harmonic_mean = []
accuracy_matrix = [[] for kk in range(args.num_tasks)]
#area_under_curve = []
overall_acc = []
replay_Classes = []
for t in range(args.num_tasks):
  if t == 15:
    args.class_epochs = 10
    args.num_epochs = 50
  elif t == 18:
    args.class_epochs = 5
    args.num_epochs = 20


  print('Task:', t + 1)
  CLASSI = Classifier.CLASSIFIER(args).to(device)
  class_appr = classifier_train(CLASSI, args)

  trainData = torch.tensor(trainData1[t], dtype = torch.float32)
  trainLabels = torch.tensor(trainLabels1[t])
  #print(trainLabels.shape, 'shape00')
  trainLabelVectors = torch.tensor(trainLabelsVectors1[t], dtype = torch.float32)


  testData = torch.tensor(testData1[t], dtype = torch.float32)
  testLabels = torch.tensor(testLabels1[t], dtype = torch.int64)
  X_train = torch.cat([trainData , trainLabelVectors], dim=1).to(args.device)

  if t == 0:
    #print(t, trainData.shape, trainLabels.shape, trainLabelVectors.shape, )
    appr.train(t, trainData, trainLabels, trainLabelVectors)
    replay_Classes = replay_Classes + sorted(list(set(trainLabels.detach().numpy().tolist())))
  else:
    replay_TrainData = []
    replay_TrainLabels = []
    replay_TrainAttr = []
    replay_Exs = len(replay_Classes) * replay_nSamples

    noise_gen = torch.randn([replay_Exs, args.n_z], dtype = torch.float32).to(args.device)

    for tc in replay_Classes:
      for ii in range(0, replay_nSamples):
        replay_TrainAttr.append(ATTR[tc])
        replay_TrainLabels.append(tc)

    replay_TrainAttr = torch.tensor(replay_TrainAttr, dtype = torch.float32).to(args.device)
    replay_TrainLabels = torch.tensor(replay_TrainLabels, dtype = torch.int64).to(args.device)

    #replay_dec_ip = torch.cat([noise_gen, replay_TrainAttr], dim = 1)
    replay_TrainData = appr.test(noise_gen, replay_TrainAttr, t - 1)

    X_replay = torch.cat([replay_TrainData , replay_TrainAttr], dim = 1)
    X_train = torch.cat([X_train, X_replay], dim = 0)
    trainLabelVectors = torch.cat([trainLabelVectors.to(args.device), replay_TrainAttr], dim=0)
    trainData = torch.cat([trainData.to(args.device), replay_TrainData], dim=0)
    #print(trainLabels.shape, 'shape1')
    trainLabels11 = torch.cat([trainLabels.to(args.device), replay_TrainLabels], dim = 0)

    appr.train(t, trainData.cpu(), trainLabels11.cpu(), trainLabelVectors.cpu())
    
    replay_Classes = replay_Classes + sorted(list(set(trainLabels.detach().numpy().tolist())))

  #test_model = appr.load_model(t)

  testLabels_seen = []
  testLabels_unseen = []
  testData_seen = []
  testData_unseen = []


  

  for ll in range(t+1):
    testData_seen = testData_seen + list(testData1[ll])
    #print(testData_seen, 'testData_seen')
    testLabels_seen = testLabels_seen + list(testLabels1[ll])
  ll = args.num_tasks - 1
  
  while True:
    if ll > t:
      testData_unseen = testData_unseen + list(testData1[ll])
      testLabels_unseen = testLabels_unseen + list(testLabels1[ll])
      ll = ll - 1
    else:
      break
  print("****************")
  print("the length of test_data_seen.....", len(testData_seen))
  print("the length of test_data_unseen.....", len(testData_unseen))
  print("***************")

  trainClasses = sorted(list(set(testLabels_seen)))
  testClasses = sorted(list(set(testLabels_unseen)))

  #print(trainClasses, 'trainClasses', testClasses)
  testData_seen111 = 1 * testData_seen
  testData_unseen111 = 1 * testData_unseen
  #added seen and unseen class to generate data for both classes
  testClasses = testClasses + trainClasses


  # ==================================================
  # generate data to train classifier
  pseudoTrainData = []
  pseudoTrainLabels =[]
  pseudoTrainAttr = []
  totalExs = len(testClasses)*nSamples

  z = torch.as_tensor(np.random.normal(0., 1., (totalExs,args.n_z)),dtype=torch.float32).to(args.device)
  
  for tc in testClasses:
    for ii in range(0,nSamples):
      pseudoTrainAttr.append(ATTR[tc])
      pseudoTrainLabels.append(tc)

  pseudoTrainAttr = torch.tensor(pseudoTrainAttr, dtype = torch.float32).to(args.device)
  pseudoTrainLabels = torch.tensor(pseudoTrainLabels, dtype = torch.int64)
  #print(pseudoTrainAttr.shape, '1', z.shape)
  #z_attr = torch.cat([z, pseudoTrainAttr], dim = 1)

  #print(testData_seen, 'testData_seen')
  share_out = appr.test(z, pseudoTrainAttr, t)
  share_out = share_out.to('cpu')
  #print(share_out, '222')
  

  #test_model.eval()

  #pseudoTrainData, _ = test_model.decoder_of_shared_and_private(z, pseudoTrainAttr, t)
  
  #CLASSI.load_state_dict(torch.load(path))
  #class_appr = classifier_train(CLASSI, args, Classifier)

  pseudoTrainData1 = torch.from_numpy(normalize(share_out.detach().numpy(), axis = 1))

  #print(pseudoTrainData1, pseudoTrainLabels)
  
  class_appr.train(t, pseudoTrainData1, pseudoTrainLabels)

  

  #test_class_model = class_appr.load_class_model(t)
  #test_class_model.eval()

  testData_seen1 = torch.from_numpy(normalize(np.array(testData_seen).astype(np.float32), axis = 1)).to(args.device)

  pred_s = class_appr.test(testData_seen1)
  #print(pred_s, 'pred_s')

  #pred_s = torch.argmax(test_class_model(testData_seen1), axis = 1)

  #print(pred_s, 'pred_s')
  allSeenClasses = sorted(list(set(testLabels_seen)))
  allUnseenClasses = sorted(list(set(testLabels_unseen)))
  dict_correct = {}
  dict_total = {}

  for ii in range(args.num_classes):
    dict_total[ii] = 0
    dict_correct[ii] = 0

  for ii in range(0,np.array(testLabels_seen).shape[0]):
    if(testLabels_seen[ii] == pred_s[ii]):
      dict_correct[testLabels_seen[ii]] = dict_correct[testLabels_seen[ii]] + 1
    dict_total[testLabels_seen[ii]] = dict_total[testLabels_seen[ii]] + 1
            
  avgAcc1 = 0.0
  avgAcc2 = 0.0
  num_seen = 0.0
  num_unseen = 0.0
    
  # seen classes accuracy for current task
  for ii in allSeenClasses:
    avgAcc1 = avgAcc1 + (dict_correct[ii]*1.0)/(dict_total[ii])
    num_seen = num_seen + 1
        
  avgAcc1 = avgAcc1/num_seen
  seen_acc.append(avgAcc1)
  print('seen_acc:', seen_acc)

  if t != args.num_tasks - 1:
    testData_unseen1 = torch.from_numpy(normalize(np.array(testData_unseen).astype(np.float32), axis = 1)).to(args.device)
  
    pred_us = class_appr.test(testData_unseen1)
    #print(pred_us.shape, 'us', torch.tensor(testLabels_unseen).shape)

    #pred_us = torch.argmax(test_class_model(testData_unseen1), axis = 1)    

    #print(pred_us, 'pred_us')
    for ii in range(0,np.array(testLabels_unseen).shape[0]):
      #print(ii, 'ii', testLabels_unseen[ii], pred_us[ii])
      if(testLabels_unseen[ii] == pred_us[ii]):
        dict_correct[testLabels_unseen[ii]] = dict_correct[testLabels_unseen[ii]] + 1
      dict_total[testLabels_unseen[ii]] = dict_total[testLabels_unseen[ii]] + 1
            
      # unseen classes accuracy for current task
    for ii in allUnseenClasses:
      avgAcc2 = avgAcc2 + (dict_correct[ii]*1.0)/(dict_total[ii])
      num_unseen = num_unseen + 1
            
    avgAcc2 = avgAcc2/num_unseen
    unseen_acc.append(avgAcc2)
    print('unseen_acc', unseen_acc)
    hm111 = (2 * seen_acc[t] * unseen_acc[t]) /(seen_acc[t] + unseen_acc[t])
    harmonic_mean.append(hm111)

    print('HM acc:', harmonic_mean)
#overall accuracy:
  testData_total = list(testData_seen111) + list(testData_unseen111)
  testData_total = torch.tensor(testData_total, dtype = torch.float32).to(args.device)
  targets = testLabels_seen + testLabels_unseen
  targets_classes = sorted(list(set(targets)))
    
  pred_ov = class_appr.test(testData_total)#classifier.predict(testData_total)
  #pred_ov = torch.argmax(pred_ov, dim = 1)
    
  dict_correct_oacc = {}
  dict_total_oacc = {}

  for ii in targets_classes:
    dict_total_oacc[ii] = 0
    dict_correct_oacc[ii] = 0

  for ii in range(0,np.array(targets).shape[0]):
    if(targets[ii] == pred_ov[ii]):
      dict_correct_oacc[targets[ii]] = dict_correct_oacc[targets[ii]] + 1
    dict_total_oacc[targets[ii]] = dict_total_oacc[targets[ii]] + 1
            
  avgAcc_ov = 0.0
  num_seen_ov = 0.0
        
  for ii in targets_classes:
    avgAcc_ov = avgAcc_ov + (dict_correct_oacc[ii]*1.0)/(dict_total_oacc[ii])
    num_seen_ov = num_seen_ov + 1
        
  avgAcc_ov = avgAcc_ov/num_seen_ov
          
  overall_acc.append(avgAcc_ov)
  print('overall_acc', overall_acc)

#########################################################################
# To compute accuracy matrix:
  for kk in range(args.num_tasks):
    testData_tw = torch.tensor(testData1[kk], dtype = torch.float32).to(args.device)
    testLabels_tw = torch.tensor(testLabels1[kk], dtype = torch.int64)
    testLabels_tw_classes = sorted(list(set(testLabels_tw.detach().numpy().tolist())))
        
    pred_tw = (class_appr.test(testData_tw)).cpu() #classifier.predict(testData_tw)
    #pred_tw = torch.argmax(pred_tw, dim = 1)        
            
    dict_correct_tw = {}
    dict_total_tw = {}

    for ii in testLabels_tw_classes:
      dict_total_tw[ii] = 0
      dict_correct_tw[ii] = 0

    for ii in range(0, testLabels_tw.shape[0]):
      if(testLabels_tw[ii] == pred_tw[ii]):
        dict_correct_tw[testLabels_tw[ii].item()] = dict_correct_tw[testLabels_tw[ii].item()] + 1
      #print(testLabels_tw[ii], '1', dict_total_tw[testLabels_tw[ii]], '2', dict_total_tw[testLabels_tw[ii]])
      dict_total_tw[testLabels_tw[ii].item()] = dict_total_tw[testLabels_tw[ii].item()] + 1
            
    avgAcc_tw = 0.0
    num_seen_tw = 0.0
        
    for ii in testLabels_tw_classes:
      avgAcc_tw = avgAcc_tw + (dict_correct_tw[ii]*1.0)/(dict_total_tw[ii])
      num_seen_tw = num_seen_tw + 1
        
      avgAcc_tw = avgAcc_tw/num_seen_tw
        
      #testData_tw[jj].append(avgAcc_tw)
      accuracy_matrix[t].append(avgAcc_tw)
  #print(accuracy_matrix, 'testData_tw')

#########################################################################

#compute average harmonic mean
for jj in range(args.num_tasks - 1):
    hm = (2 * (seen_acc[jj] * unseen_acc[jj]))/(seen_acc[jj] + unseen_acc[jj])
    harmonic_mean.append(hm)
print("the harmonic mean is...", np.mean(np.array(harmonic_mean)))
print("the seen acc mean is...", np.mean(np.array(seen_acc)))
print("the unseen acc mean is...", np.mean(np.array(unseen_acc)))

#calculating forgetting measure:
accuracy_matrix = np.array(accuracy_matrix)
forgetting_measure = []
for after_task_idx in range(1, args.num_tasks):
    after_task_num = after_task_idx + 1
    prev_acc = accuracy_matrix[:after_task_num - 1, :after_task_num - 1]
    forgettings = prev_acc.max(axis=0) - accuracy_matrix[after_task_num - 1, :after_task_num - 1]
    forgetting_measure.append(np.mean(forgettings).item())
    
print('forgetting_measure', forgetting_measure)
print("the forgetting measure is...", np.mean(np.array(forgetting_measure)))


#calculating joint accuracy:
mean_joint_acc = np.mean(np.array(overall_acc))
print(mean_joint_acc, 'mean_overall_acc')
  






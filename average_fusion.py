from matplotlib import pyplot as plt
import pickle
import numpy as np
import torch
from utils import *
import dataloader
from sklearn.metrics import confusion_matrix
import itertools

# for computing confusion matrix
def compute_confusion_matrix(y_actu, y_pred, class_names):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_actu, y_pred)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    #uncomment below lines if you want to plot non-normalized matrix
    #plt.figure()
    #plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure(figsize=(20,20))
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

    plt.show()


#for plotting confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




if __name__ == '__main__':

    rgb_preds='record/spatial/spatial_video_preds.pickle'
    #opf_preds = 'record/motion/motion_video_preds.pickle'
    opf_preds = 'record/motion_pose/motion_pose_video_preds.pickle'

    with open(rgb_preds,'rb') as f:
        rgb =pickle.load(f)
    f.close()
    with open(opf_preds,'rb') as f:
        opf =pickle.load(f)
    f.close()

    dataloader = dataloader.spatial_dataloader(BATCH_SIZE=1, num_workers=1,
                                    path='/home/molly/UCF_data/jpegs_256',
                                    ucf_list='/home/molly/two-stream-action-recognition/UCF_list/',
                                    ucf_split='04')
    train_loader,val_loader,test_video = dataloader.run()

    video_level_preds = np.zeros((len(rgb.keys()),15))
    video_level_labels = np.zeros(len(rgb.keys()))
    correct=0
    ii=0
    for name in sorted(rgb.keys()):
        r = rgb[name]
        o = opf[name]

        label = int(test_video[name])-1

        video_level_preds[ii,:] = (r+o)
        video_level_labels[ii] = label
        ii+=1
        if np.argmax(r+o) == (label):
            correct+=1

    video_level_labels = torch.from_numpy(video_level_labels).long()
    video_level_preds = torch.from_numpy(video_level_preds).float()

    top1,top5 = accuracy(video_level_preds, video_level_labels, topk=(1,3))

    class_names = ['Archery','BaseballPitch','BodyWeightSquats','CliffDiving',
                   'Diving','FloorGymnastics','GolfSwing','JumpingJack','JumpRope',
                   'Lunges','MoppingFloor','PullUps','Skiing','TaiChi','WallPushups']

    #print(torch.argmax(video_level_preds,axis = 1))
    #print(video_level_labels)

    compute_confusion_matrix(video_level_labels, torch.argmax(video_level_preds,axis = 1), class_names)

    print(top1, top5)

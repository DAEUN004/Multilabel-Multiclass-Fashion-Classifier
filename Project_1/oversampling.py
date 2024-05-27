import numpy as np
import time

def IRLbl(labels):
    # import one hot encoded label
    pos_nums_per_label = np.sum(labels, axis=0)
    max_pos_nums = np.max(pos_nums_per_label)
    return max_pos_nums / pos_nums_per_label

def MeanIR(labels):
    IRLbl_VALUE = IRLbl(labels)
    return np.mean(IRLbl_VALUE)


def ML_ROS(all_labels, indices=None, num_samples=None, Preset_MeanIR_value=1.,
                 max_clone_percentage=80, sample_size=32):
    # the index of samples: 0, 1, ....
    # if indices is not provided,
    # all elements in the dataset will be considered
    indices = list(range(len(all_labels))) \
        if indices is None else indices

    # if num_samples is not provided,
    # draw `len(indices)` samples in each iteration
    num_samples = len(indices) \
        if num_samples is None else num_samples

    MeanIR_value = MeanIR(all_labels) if Preset_MeanIR_value == 0 else Preset_MeanIR_value
    IRLbl_value = IRLbl(all_labels)
    # N is the number of samples, C is the number of labels
    N, C = all_labels.shape
    # the samples index of every class
    indices_per_class = {}
    minority_classes = []
    # accroding to psedu code, maxSamplesToClone is the upper limit of the number of samples can be copied from original dataset
    maxSamplesToClone = N / 100 * max_clone_percentage
    print('Max Clone Limit:', maxSamplesToClone)
    for i in range(C):
        ids = all_labels[:, i] == 1
        # How many samples are there for each label
        indices_per_class[i] = [ii for ii, x in enumerate(ids) if x]
        if IRLbl_value[i] > MeanIR_value:
            minority_classes.append(i)

    new_all_labels = all_labels
    oversampled_ids = []
    minorNum = len(minority_classes)
    print(minorNum, 'minor classes.')

    for idx, i in enumerate(minority_classes):
        tid = time.time()
        while True:
            pick_id = list(np.random.choice(indices_per_class[i], sample_size))
            indices_per_class[i].extend(pick_id)
            # recalculate the IRLbl_value
            # The original label matrix (New_ all_ Labels) and randomly selected label matrix (all_ labels[pick_ ID) and recalculate the irlbl
            new_all_labels = np.concatenate([new_all_labels, all_labels[pick_id]], axis=0)
            oversampled_ids.extend(pick_id)

            newIrlbl = IRLbl(new_all_labels)




            if newIrlbl[i] <= MeanIR_value:
                print('\nMeanIR satisfied.', newIrlbl[i])
                break
            if len(oversampled_ids) >= maxSamplesToClone:
                print('\nExceed max clone.', len(oversampled_ids))
                break
            # if IRLbl(new_all_labels)[i] <= MeanIR_value or len(oversampled_ids) >= maxSamplesToClone:
            #     break
            print("\roversample length:{}".format(len(oversampled_ids)), end='')
        print('Processed the %d/%d minor class:' % (idx+1, minorNum), i, time.time()-tid, 's')
        if len(oversampled_ids) >= maxSamplesToClone:
            print('Exceed max clone. Exit', len(oversampled_ids))
            break
    return new_all_labels, oversampled_ids



def get_all_labels(root, annFile, ):
    import os
    import torch

    NUM_ATTR = 6
    NUM_CLASSES_PER_CATEGORY = [7, 3, 3, 4, 6, 3]
    img_path = []
    labels = [[] for _ in range(6)]

    with open(annFile) as f:
        for line in f:
            img_path.append(os.path.join(root, line.split()[0]))
        with open(annFile.replace('.txt', '_attr.txt')) as f:
                for line in f:
                    attrs = line.split()
                    for i in range(6):
                        labels[i].append(int(attrs[i]))
    one_hot_labels = []
    for i in range(NUM_ATTR):
        num_classes = NUM_CLASSES_PER_CATEGORY[i]
        one_hot_label = torch.nn.functional.one_hot(torch.tensor(labels[i]), num_classes=num_classes)
        one_hot_labels.append(one_hot_label)
    
    one_hot_labels = torch.cat(one_hot_labels, dim=1)
    labels = np.array(one_hot_labels)
    print('All labels in the dataset:', labels.shape)

    return labels

def copy_samples(root, annFile, newAnnFile, newAnnFile_2, copyIds):
    import os
    from shutil import copyfile
    import sys

    # adding exception handling
    srcFile = os.path.join(root, annFile)
    dstFile = os.path.join(root, newAnnFile)
    dstFile2 = os.path.join(root, newAnnFile_2)
    # copy original file to new file
    try:
        copyfile(srcFile, dstFile)
        copyfile('FashionDataset/FashionDataset/split/train_attr.txt', dstFile2)
    except IOError as e:
        print("Unable to copy file. %s" % e)
        exit(1)
    except:
        print("Unexpected error:", sys.exc_info())
        exit(1)

    # copy samples to new file
    copied = 0
    txt = np.loadtxt(srcFile, dtype=str)
    copyNum = len(copyIds)
    txt = txt.tolist()

    for lineId in copyIds:
        newAnnFile = open(dstFile, 'a')
        line = txt[lineId]
        line = ''.join(line) + '\n'
        newAnnFile.write(line)
        copied += 1
        print('\rCopied %d/%d' % (copied, copyNum), end='')

    copied = 0
    txt2 = np.loadtxt('FashionDataset/FashionDataset/split/train_attr.txt', dtype=str)
    txt2 = txt2.tolist()

    for lineId in copyIds:
        newAnnFile_2 = open(dstFile2, 'a')
        line = txt2[lineId]
        line = ' '.join(line) + '\n'
        newAnnFile_2.write(line)
        copied += 1
        print('\rCopied %d/%d' % (copied, copyNum), end='')

    
    print('\nUpsampling Done. ', dstFile)

# use ml-ros to process celeba dataset, reduce the class imbalance
def mlros_celeba():
    celebaRoot = ""
    annFile = "FashionDataset/FashionDataset/split/train.txt"
    allLabels = get_all_labels(celebaRoot, annFile,)


    print('Origianl:', len(allLabels))
    t1 = time.time()
    newLables, oversampleIds = ML_ROS(allLabels, )
    print('New:', len(newLables), time.time()-t1, 's')
    # generate new train.txt
    newAnnFile = annFile.replace('.txt', '_oversampled.txt')
    newAnnFile2 = "FashionDataset/FashionDataset/split/train_attr.txt".replace('.txt', '_oversampled.txt')
    copy_samples(celebaRoot, annFile, newAnnFile, newAnnFile2, oversampleIds)

mlros_celeba()

###test_data = FashionNet_Dataset("FashionDataset/FashionDataset","FashionDataset/FashionDataset/split/test.txt")
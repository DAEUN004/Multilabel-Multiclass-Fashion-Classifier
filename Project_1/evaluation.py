# evaluation
import numpy as np
import torch 
def compute_avg_class_acc(gt_labels, pred_labels):
    num_attr = 6
    num_classes = [7, 3, 3, 4, 6, 3]  # number of classes in each attribute

    per_class_acc = []
    #attr_ix = 0~6
    for attr_idx in range(num_attr):
        #First attribute: 7 --> 0 1 2 3 4 5 6
        for idx in range(num_classes[attr_idx]):
            target = gt_labels[:, attr_idx]
            pred = pred_labels[:, attr_idx]
            correct = torch.sum((target == pred) * (target == idx))
            total = torch.sum(target == idx)

            # Avoid division by zero
            if total != 0:
                per_class_acc.append(float(correct) / float(total))
            else:
                per_class_acc.append(0.0)  # Append zero when total is zero

    # Avoid division by zero when calculating the average
    if len(per_class_acc) != 0:
        return sum(per_class_acc) / len(per_class_acc)
    else:
        return 0.0
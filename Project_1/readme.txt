○ evaluation.py: Computes the average class accuracy given ground truth labels and predicted labels. 

○ load_data.py: Loads image paths and associated labels from text files, preprocesses images using torchvision transforms, and converts labels to one-hot encoding. 

○ main.py: This file trains a fashion attribute recognition model using VGG16 architecture saves loss/accuracy plots, predicts on the test set, and saves the trained model

○ oversampling.py: provides functions for handling class imbalance in multi-label datasets using oversampling. (run separately to create separate new text files)

○ utils.py: This file contains functions to convert multi-label data into argmax values and plot loss and accuracy during model training. 
            Functions include converter for data conversion and plot_loss, plot_acc, and plot_loss_acc for plotting loss and accuracy over epochs.


○ References to the third-party libraries: for oversampling.py refer to https://github.com/Light--/ML-ROS-multi-label-random-oversampling.git

cd /home/msai/daeun004/Project_1

○ How to run the script to produce the result: 

    > job.sh

    python main.py \
    --epochs 100 \
    --fig_name figures.png \
    --test

○ Run with model checkpoint:

    > job.sh

    python get_test.py
    

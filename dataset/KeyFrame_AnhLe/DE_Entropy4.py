import os, sys
import shutil

import random
import cv2
import math
import imageio
from sklearn.metrics.cluster import entropy
from dataset.csv_lib import writefile_csv_dic

# MAX_NUMBER_OF_FRAMES = 100 # numFrames

# TOTAL_KEY_FRAMES = 5  # numKeyFrames

# STOPPING_ITERATION = 40  # numGeneration

# NUMBER_OF_NP_CANDIDATES = 10  #popSize

# Location to read images from.
# NOTE: "_" is used to follow a naming convention
# eg. _20.jpg, _21.jpg etc...
# source = "/content/gdrive/MyDrive/Colab Notebooks/2020-MNADrc/2020-MNADrc/dataset/ped2/training/frames/01/"

# Location to write GIF images to.
# dest = "/content/gdrive/MyDrive/Colab Notebooks/2020-MNADrc/2020-MNADrc/dataset/ped2/training/frames/sample.GIF"

# Population matrix.
NP = []

# Mutation vector.
MV = []

# Trail vector.
TV = []

# Scale factor.
F = 0.9

# Cr probability value.
Cr = 0.6


# Calculate Average Entropy Difference for a chromosome.
def getEntropy(source, numKeyFrames, KF):
    entropy_sum = 0
    for i in range(0, numKeyFrames - 1):
        while True:
            try:
                im1 = cv2.imread(os.path.join(source, '{:04d}.jpg'.format(KF[i]+1)), 0)
                im2 = cv2.imread(os.path.join(source, '{:04d}.jpg'.format(KF[i + 1]+1)), 0)
                entropy_sum += abs(entropy(im1) - entropy(im2))
            except:
                #  print (i, KF, KF[i], KF[i+1])
                continue
            break
    return entropy_sum / numKeyFrames


# INITIALISATION : Generates population NP of 10 parent vectors (and Average Entropy Differences).
def initialize_NP(source, numFrames, numKeyFrames, popSize):
    print('Number of frame initialize_NP=', numFrames)
    print('numKeyFrames initialize_NP=', numKeyFrames)
    global NP
    NP = []
    # Mutation vector.
    global MV
    MV = []
    # Trail vector.
    global TV
    TV = []
    # Scale factor.
    global F
    F = 0.9
    # Cr probability value.
    global Cr
    Cr = 0.6
    for i in range(popSize):
        NP.append(sorted(random.sample(range(0, numFrames - 1), numKeyFrames)))
        NP[-1].append(getEntropy(source, numKeyFrames, NP[-1]))
        # print ('NP=',NP)
        # print ('NP[-1]=', NP[-1])


# MUTATION
def mutation(source, numFrames, numKeyFrames, parent):
    R = random.sample(NP, 2)
    global MV
    MV[:] = []
    MV_value = 0
    #  print ('parent=',parent)
    #   print ('NP[parent]=', NP[parent])
    for i in range(numKeyFrames):
        MV_value = int(NP[parent][i] + F * (R[0][i] - R[1][i]))
        if MV_value < 0:
            MV.append(0)
        elif MV_value >= numFrames:
            MV.append(numFrames - 1)
        else:
            MV.append(MV_value)
    MV.sort()
    MV.append(getEntropy(source, numKeyFrames, MV))


#  print ('MV=', MV)

# CROSSOVER (uniform crossover with Cr = 0.6).
def crossover(source, numKeyFrames, parent, mutant):
    # print ("mutant: ", mutant)
    # print ("parent: ", parent)
    for j in range(numKeyFrames):
        if random.uniform(0, 1) < Cr:
            TV.append(mutant[j])
        else:
            TV.append(parent[j])
    TV.sort()
    TV.append(getEntropy(source, numKeyFrames, TV))


# print ("TV    : ", TV)

# SELECTION : Selects offspring / parent based on higher Entropy diff. value.
def selection(parent, trail_vector):
    if trail_vector[-1] > parent[-1]:
        parent[:] = trail_vector
    #    print ("yes", parent)


# else:
#     print ("no")

# bestParent returns the parent with then maximum Entropy diff. value.
def bestParent(population):
    Max_Entropy_value = population[0][-1]
    Best_Parent_Index = population[0]
    for parent in population:
        if parent[-1] > Max_Entropy_value:
            Max_Entropy_value = parent[-1]
            Best_Parent_Index = parent
    return Best_Parent_Index


def evolution4(source, numGeneration, popSize, numFrames, numKeyFrames, keyframe_dir, path2file):
    """
    Evolution for a video
    param numGeneration: Max number of generation
    param popSize: the number of candidates
    param numFrames: the total of frames
    param numKeyFrames: the number of key frames
    """
    # 1. Initialize
    initialize_NP(source, numFrames, numKeyFrames, popSize)

    # 2. For each Generation
    for GENERATION in range(numGeneration):
        for i in range(popSize):
            # print("---------------------", "PARENT:", i + 1, "GENERATION:", GENERATION + 1, "---------------------")
            mutation(source, numFrames, numKeyFrames, i)
            crossover(source, numKeyFrames, NP[i], MV)
            selection(NP[i], TV)
            # print(NP[i])
            TV[:] = []
            # print(" ")
        # print(" ")
    best_parent = bestParent(NP)
    print("best solution is: ", best_parent)

    # images_for_gif = []
    for frame_number in best_parent[:-1]:
        keyframe = os.path.join(source, '{:04d}.jpg'.format(frame_number+1))
        # images_for_gif.append(imageio.imread(keyframe))
        file_name = keyframe.split('/')[-1]
        dest_file = os.path.join(keyframe_dir, file_name)
        shutil.copy(keyframe, dest_file)

        fieldnames = ['id', 'name']
        row = {'id': str(frame_number), 'name': file_name}
        if frame_number == 0:
            writefile_csv_dic(path2file, 'w', fieldnames, row)
        else:
            writefile_csv_dic(path2file, 'a', fieldnames, row)

    # imageio.mimsave(keyframe_dir + '/sample.GIF', images_for_gif)
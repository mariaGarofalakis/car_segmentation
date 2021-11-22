import os
from os import path
import shutil

#src = "C:/Master_DTU/4th semester/"
#parent_dir = "C:/Users/george/Desktop/DL_project"

# ARGUMENTS:
#            1) source_directory = the path of the folder of the mixed data ( in my case src -see above-)
#            2) destination_directory = the path of the folder where we create the 2 directories (trainset,testset)
#            3) parent_dir_txt = the path of the folder of the txt files 


#  Returns:   train_files = Names of training files 
#             fake_files = Names of fake files 
#             test_files = Names of test files 
#             train_path
#             fake path
#             test_path


def divide_train_test_set(source_directory, destination_directory, parent_dir_txt):
    #Names of txt files
    test_txt = "test.txt"
    non_train_txt = "non_train.txt"
#create new directories to move training and test data
    train_directory_name = "trainset"
    test_directory_name = "testset"
    fake_dir = "fake_images"
  

    train_path = os.path.join(destination_directory, train_directory_name)
    test_path = os.path.join(destination_directory, test_directory_name)
    fake_path = os.path.join(destination_directory, fake_dir)
    
    non_train_txt_path = os.path.join( parent_dir_txt, non_train_txt)
    test_txt_path = os.path.join( parent_dir_txt, test_txt)
    
    # Read Names of npy files at the txt files
    with open(non_train_txt_path, 'r') as f:
        non_train_Names = f.read().splitlines()
        
    with open(test_txt_path, 'r') as f:
        test_Names = f.read().splitlines()

    os.mkdir(train_path)
    os.mkdir(test_path)
    os.mkdir(fake_path)

#move data to train directory, fake and test directory respectively
    train_files = [i for i in os.listdir(source_directory) if i not in test_Names and 
                   'aug' not in i and path.isfile(path.join(source_directory, i)) and 
                   i not in non_train_Names and
                   "OPEL" not in i and "DOOR" not in i]
    
    for f in train_files:
        shutil.move(path.join(source_directory, f), train_path)
        
        
    fake_files = [i for i in os.listdir(source_directory) if "OPEL" in i or "DOOR" in i]
    for f in fake_files:
        shutil.move(path.join(source_directory, f), fake_path)



    test_files = test_Names
    for f in test_files:
        shutil.move(path.join(source_directory, f), test_path)

    return train_files, fake_files, test_files, train_path, fake_path, test_path
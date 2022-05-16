import os
import glob
import shutil


target_folder = './tiny-imagenet-200/val/'

val_dict = {}
with open('./tiny-imagenet-200/val/val_annotations.txt', mode='r') as f:
    for line in f.readlines():
        line = line.split('\t')
        val_dict[line[0]] = line[1]

paths = glob.glob('./tiny-imagenet-200/val/images/*')
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    if not os.path.exists(target_folder + str(folder)):
        os.mkdir(target_folder + str(folder))
        os.mkdir(target_folder + str(folder) + '/images')

for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    dest = target_folder + str(folder) + '/images/' + str(file)
    shutil.move(path, dest)

os.rmdir('./tiny-imagenet-200/val/images')
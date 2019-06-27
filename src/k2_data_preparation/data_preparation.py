# Divide the dataset into train, validation and test
import os
import shutil
import logging

logger = logging.getLogger("Data Preparation")

original_dataset_dir = 'data/src'
classes = os.listdir(original_dataset_dir)
subdirectories = ['train', 'validation', 'test']
subdirectories_len = {'train':2/3, 'validation': 1/6, 'test': 1/6}

base_dir = 'data/raw'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# Create directories
for subdirectory in subdirectories:
    sub_dir = os.path.join(base_dir, subdirectory)
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)

directories = os.listdir(base_dir)

for sub_dir in directories:
    for class_ in classes:
        sub_class_dir = os.path.join(sub_dir, class_)
        if not os.path.exists(sub_class_dir):
            os.mkdir(sub_class_dir)

for class_ in classes:
    origin_dir = os.path.join(original_dataset_dir, class_)
    for r, d, f in os.walk(origin_dir):
        jpg_files = [file for file in f if '.jpg' in file]
        prev_number = 1
        for sub_dir in subdirectories:
            dest_dir = os.path.join(base_dir, sub_dir, class_)
            img_number = int(subdirectories_len.get(sub_dir)*len(jpg_files))
            next_number = prev_number + img_number
            fnames = ['{}{}.jpg'.format(class_, i) for i in range(
                prev_number, next_number)]
            prev_number = next_number
            for fname in fnames:
                src = os.path.join(origin_dir, fname)
                dst = os.path.join(dest_dir, fname)
                shutil.copyfile(src, dst)
            logger.info('{} images have been successfully copied from {} to {}'.format(img_number, origin_dir, dest_dir))
# This file is used to create gt.txt for dataset to load
import os
import glob
import fire

def create_gt(image_folder,gt_folder,outfile='./gt.txt'):
    '''
    Create the groundtruth txt file
    :param image_folder:
    :param gt_folder:
    :param outfile:
    :return:
    '''
    with open(outfile, 'w') as f:
        for imagepath in glob.glob(os.path.join(image_folder, "*.*")):
            imagepathroot, (imagename, imageext) = os.path.split(imagepath)[0], os.path.splitext(os.path.split(imagepath)[1])
            gtpath = glob.glob(os.path.join(gt_folder, f"{imagename}.*"))
            if len(gtpath)>0:
                gtpath = gtpath[0]
            else:
                continue

            f.write(f"{imagepath};{gtpath}\n")


if __name__ == "__main__":
    fire.Fire(create_gt)


from os import listdir
from os.path import isfile, join
import cv2
import os
from services.get_uid_from_image import get_uids
from services.train.uid_finder_dl import find_uid
from services.train.seed_counter_dl import find_seeds

if __name__ == "__main__":
    get_uids("D:/python_projects/pheno-backend/assets")
    uids = find_uid("D:/python_projects/pheno-backend/results/uid_crops")
    seeds = find_seeds("D:/python_projects/pheno-backend/assets")
    print(list(zip(uids, seeds)))

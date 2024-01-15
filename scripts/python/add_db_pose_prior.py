#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  REALCAT, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  REALCAT, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Vincent Qin
#  Creation Date      : 2021.05.12
#  Description        : add image pose prior to database
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import os
import sys
import numpy as np
from tqdm import tqdm
from database import COLMAPDatabase
from read_write_model import read_model
from pathlib import Path
from shutil import copyfile

FORMAT_LISTS = [".png", ".jpeg", ".jpg", ".bmp"]


def get_prior_poses(images: dict) -> tuple[dict, dict]:
    """
    Generates prior poses for a given set of images.

    Args:
        images (dict): A dictionary of images.

    Returns:
        tuple[dict, dict]: A tuple containing two dictionaries. The first dictionary contains
        the prior poses for each image, where the image name is the key and the corresponding
        translation vector is the value. The second dictionary contains the prior poses for
        each image, where the image name is the key and the corresponding quaternion vector
        is the value.
    """
    t_prior_poses = {}
    q_prior_poses = {}

    for item in tqdm(images.items()):
        image = item[1]
        image_name = image.name
        q_cw = image.qvec  # qw qx qy qz
        t_cw = image.tvec
        t_prior_poses[image_name] = t_cw
        q_prior_poses[image_name] = q_cw
    return t_prior_poses, q_prior_poses


def main(path_db, path_prior_sfm_model, path_images):
    """
    Update the prior poses of images in the database.

    Args:
        path_to_db (str): Path to the COLMAP database file.
        path_to_poses (str): Path to the poses text file.
        path_to_images (str, optional): Path to the images directory. Defaults to None.
    """
    # Check if the database and poses file exist
    if not os.path.exists(path_db) and not os.path.exists(path_prior_sfm_model):
        print(
            "ERROR: database path don't exists:{} or prior_sfm_model path don't exists:{}\n".format(
                path_db, path_prior_sfm_model
            )
        )
        return

    path_db_new = path_db.replace(".db", "_new.db")
    copyfile(path_db, path_db_new)

    # Read images from poses file
    _, images, _ = read_model(path_prior_sfm_model)  # prior pose!

    # Get prior poses for each image
    t_prior_poses, q_prior_poses = get_prior_poses(images)

    # Open the database
    db = COLMAPDatabase.connect(path_db_new)

    # Read images from the database
    rows = db.execute("SELECT * FROM images")

    # Get the list of images in the images directory
    images_list = os.listdir(path_images)

    # Update the prior poses for each image in the database
    for item in tqdm(images_list):
        if Path(item).suffix not in FORMAT_LISTS:
            continue
        image = next(rows)

        # Initialize prior translation with NaN values
        prior_t = np.full(3, np.NaN)
        prior_q = np.full(4, np.NaN)

        image_id = image[0]
        name = image[1]
        camera_id = image[2]

        # Check if the image name has prior poses
        if name in t_prior_poses.keys():
            prior_t = t_prior_poses[name]
            prior_q = q_prior_poses[name]

        # Update the image's prior poses in the database
        sql_update = "UPDATE images SET name='%s',camera_id=%d,\
                      prior_qw='%f',prior_qx='%f',prior_qy='%f',\
                      prior_qz='%f',prior_tx='%f',prior_ty='%f',\
                      prior_tz='%f' WHERE image_id=%d"
        db.execute(
            sql_update
            % (
                name,
                camera_id,
                prior_q[0],
                prior_q[1],
                prior_q[2],
                prior_q[3],
                prior_t[0],
                prior_t[1],
                prior_t[2],
                image_id,
            )
        )

    # Commit the changes and close the database
    db.commit()
    db.close()
    print(f"Add pose prior done! New database: {path_db_new}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_db", required=True, default="path/to/database.db"
    )
    parser.add_argument(
        "--path_prior_sfm_model",
        required=True,
        default="path/to/prior_sfm_model_folder",
    )
    parser.add_argument(
        "--path_images", required=True, default="path/to/images_folder"
    )
    args = parser.parse_args()
    main(**args.__dict__)

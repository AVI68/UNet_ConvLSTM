# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 17:53:06 2024

@author: Avijit Majhi
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import utility

def sort_images(image_path, prctl, thrshld):
    try:
        img_DN = utility.read_image(image_path)
        R = utility.conversion(utility.conversion(utility.conversion(img_DN, "DN_DBZ"), "DBZ_Z"), "Z_R")
        total_pixels = R.size
        R_pixels = np.count_nonzero(R > thrshld)
        img_name = os.path.basename(image_path).split(".")[0]
        p_pixl = (R_pixels / total_pixels) * 100
        if p_pixl >= prctl:
            isw_score = utility.isw_score(R)
            date = utility.fname2dt(img_name)
            return date, isw_score, p_pixl
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_images(image_paths, prctl, thrshld):
    results = []
    for image_path in tqdm(image_paths):
        result = sort_images(image_path, prctl, thrshld)
        if result is not None:
            results.append(result)
    return results

if __name__ == '__main__':
    image_folder = "D:\\Radar_data\\UNICA-Nowcasting\\SAMPLE_DATA"
    output_folder = "D:\\Radar_data\\UNICA-Nowcasting\\final_code"
    output_file = os.path.join(output_folder, "image_isw_scores.xlsx")

    prctl = 5
    thrshld = 0.1  # mm/h

    image_paths = utility.filelist(image_folder)

    # Process images sequentially
    results = process_images(image_paths, prctl, thrshld)

    if results:
        df = pd.DataFrame(results, columns=["Date", "ISW Score","Pixel_%"])
        df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No valid results were processed.")


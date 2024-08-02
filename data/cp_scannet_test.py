import shutil
import os

obj_list = ["0000_00", "0000_04", "0000_06", "0000_08", "0000_09", "0000_10", "0000_13",
             "0000_15", "0000_25", "0000_30"]
            
            

def copy_file(source_path, destination_path):
    
    shutil.copy(source_path, destination_path)
    print(f"File copied successfully from {source_path} to {destination_path}")


for obj_id in obj_list:
    obj_id=f"scene{obj_id}"
    source_file =  "scannet_origin/scans/"+obj_id+"/"+obj_id+"_vh_clean_2.ply"
    target_file = "scannet/"+obj_id+"_vh_clean_2.ply"

    copy_file(source_file, target_file)

    source_file =  "scannet_origin/scans/"+obj_id+"/"+obj_id+"_vh_clean.aggregation.json"
    target_file = "scannet/"+obj_id+"_vh_clean.aggregation.json"

    copy_file(source_file, target_file)

    source_file =  "scannet_origin/scans/"+obj_id+"/"+obj_id+"_vh_clean_2.0.010000.segs.json"
    target_file = "scannet/"+obj_id+"_vh_clean_2.0.010000.segs.json"

    copy_file(source_file, target_file)
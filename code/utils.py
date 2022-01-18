import os
import glob
import numpy as np
from tqdm import tqdm


def scan_directory(directory, ids_filter=[], filter_by_ids=False, file_rgb_posfix="_rgb", file_flow_posfix="_flow", file_extension=".npy", use_flow=False, start=0, end=10000000):

    if directory[-1] == "/":
        directory = directory[0:len(directory)-1]

    rgb_files_ = glob.glob(directory+"/*"+file_rgb_posfix+file_extension)
    rgb_files_ = sorted(rgb_files_)
    rgb_files_ = rgb_files_[start:end]
    if use_flow:
        flow_files_ = glob.glob(directory+"/*"+file_flow_posfix+file_extension)    
        flow_files_ = sorted(flow_files_)  
        flow_files_ = flow_files_[start:end]
    else:
        flow_files_ = []
    
    ids = []
    rgb_files = []
    flow_files = []
    for i in tqdm(range(len(rgb_files_))):    
        
        id = rgb_files_[i].replace(file_rgb_posfix+".",".").replace(file_extension,"").split("/")[-1]
        if use_flow:
            if rgb_files_[i].replace(file_rgb_posfix+".",".").replace(file_extension,"") != flow_files_[i].replace(file_flow_posfix+".",".").replace(file_extension,""):
                raise Exception(f"{i} - {id} - {rgb_files_[i]} and {flow_files_[i]} are missaligned")

        if filter_by_ids:
            if id in ids_filter:
                ids.append(id)
                rgb_files.append(rgb_files_[i].split("/")[-1])
                if use_flow:
                    flow_files.append(flow_files_[i].split("/")[-1])            
        else:
            ids.append(id)
            rgb_files.append(rgb_files_[i].split("/")[-1])
            if use_flow:
                flow_files.append(flow_files_[i].split("/")[-1])

    return ids, rgb_files, flow_files


def is_file_in_ids(file, ids):
    for i in ids:
        if i in file:
            return True
    return False


def load_feature_data(ids=[], directory="/", file_rgb_posfix="_rgb", 
                      file_flow_posfix="_flow", file_extension=".npy", 
                      use_flow=False, start=0, end=10000000):
    
    if ids == None or len(ids) == 0:
        filter_by_ids = False
    else:
        filter_by_ids = True    
    _, f_rgb, f_flow = scan_directory(directory=directory, 
                                        ids_filter=ids,
                                        filter_by_ids=filter_by_ids,
                                        file_rgb_posfix=file_rgb_posfix, 
                                        file_flow_posfix=file_flow_posfix, 
                                        file_extension=file_extension, 
                                        use_flow=use_flow,
                                        start=start, end=end)

    if directory[-1] == "/":
        directory = directory[0:len(directory)-1]

    # effectively loaded
    ids_loaded = []
    files_rgb = [] 
    files_flow = []

    stack_rgb = []
    stack_flow = []
    for x, i_rgb in enumerate(tqdm(f_rgb)):        
        try:
            i_flow = None
            if len(ids) != 0 and not is_file_in_ids(i_rgb, ids):
                continue
            video_stack_rgb = np.load(os.path.join(directory, i_rgb), allow_pickle=True)
            if use_flow:
                i_flow = f_flow[x] 
                if len(ids) != 0 and not is_file_in_ids(i_flow, ids):
                    continue
                video_stack_flow = np.load(os.path.join(directory,i_flow), allow_pickle=True)
            
            if not use_flow:
                if len(video_stack_rgb.shape) == 2:
                    ids_loaded.append(ids[x])
                    stack_rgb.append(video_stack_rgb) 
                    files_rgb.append(i_rgb.split("/")[-1])   
            elif len(video_stack_rgb.shape) == 2 and len(video_stack_flow.shape) == 2:
                ids_loaded.append(ids[x])
                stack_rgb.append(video_stack_rgb)
                stack_flow.append(video_stack_flow)
                files_rgb.append(i_rgb.split("/")[-1])
                files_flow.append(i_flow.split("/")[-1])
            else:
                continue
        except Exception as e:
            print(e)
            continue
    
    if len(stack_rgb) == 0:
        raise Exception("Features not loaded.")
    else:
        return ids, stack_rgb, files_rgb, stack_flow, files_flow

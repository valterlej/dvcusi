import os
import numpy as np
import time
import pickle
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def mini_batch_k_means_clustering(inp_data, output_file="./data/cluster.pkl", epochs=5, n_clusters=1000, random_state=0, batch_size=20000, save_file=True):
    
    cluster = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=batch_size)
    for e in tqdm(range(epochs)):
        start = time.time()
        print(f"Training epoch: {e}")
        np.random.shuffle(inp_data)
        cluster.fit_predict(inp_data)
        print(f"Time taken for training once {time.time()-start} sec")
        if save_file:
            print(f"Saving model")
            pickle.dump(cluster, open(output_file, 'wb'))
    return cluster


def predict(vid_stack, cluster, output_file=None, save_file=True, return_prediction=False):    
    pred = cluster.predict(vid_stack)
    pred_list = list(pred)
    if save_file:
        with open(output_file, 'wb') as f:
            np.save(f, pred)
    if return_prediction:
        return pred_list

def predict_files_from_directory(vid_stacks, files, cluster, output_dir, file_extension):
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(len(vid_stacks))):
        stack = vid_stacks[i]
        file = files[i]
        out_file = os.path.join(output_dir, file.replace("_rgb","").replace("_flow","").replace(".npy","")+file_extension)
        predict(stack, cluster, out_file, save_file=True)
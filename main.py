import argparse
import json
import numpy as np
import pickle
import time
import torch
from code.clustering import mini_batch_k_means_clustering, predict_files_from_directory
from code.config import Config
from code.utils import scan_directory, load_feature_data
from code.visualglove import predict_visual_embedding, training, VisualGloveModel
from pprint import pprint


def cluster_predictions(cfg):
    
    cluster = pickle.load(open(cfg.c_file, "rb"))    
    
    ids_directory, _, _ = scan_directory(cfg.visual_features_dir, file_rgb_posfix=cfg.file_rgb_posfix, 
                               file_flow_posfix=cfg.file_flow_posfix, 
                               file_extension=cfg.visual_file_extension, use_flow=cfg.use_flow)    
    i = 0
    n_files = 10000
    while i + n_files <= len(ids_directory) + n_files:
        print(f"Loading files from {cfg.visual_features_dir} - Batch {i}-{i+n_files}")    
        _, data_rgb, files_rgb, data_flow, _ = load_feature_data(ids_directory, directory=cfg.visual_features_dir, 
                                                                   file_rgb_posfix=cfg.file_rgb_posfix, 
                                                                   file_flow_posfix=cfg.file_flow_posfix, 
                                                                   file_extension=cfg.visual_file_extension, 
                                                                   use_flow=cfg.use_flow, 
                                                                   start=i, end=i+n_files)        
        print(f"Predicting clusters")
        if cfg.use_flow:
            data_rgb = data_rgb + data_flow
        predict_files_from_directory(data_rgb, files_rgb, cluster, cfg.output_cluster_predictions_dir, cfg.clusters_file_extension)    
        i = i + n_files


def visual_glove_embeddings(cfg):
    
    cluster = pickle.load(open(cfg.c_file, "rb"))
    glove = VisualGloveModel(cfg.vocabulary_size, cfg.vg_emb_dim)
    glove.load_state_dict(torch.load(cfg.vg_file))
    glove.to(cfg.device)

    ids_directory, _, _ = scan_directory(cfg.visual_features_dir, file_rgb_posfix=cfg.file_rgb_posfix, 
                               file_flow_posfix=cfg.file_flow_posfix, 
                               file_extension=cfg.visual_file_extension, use_flow=cfg.use_flow_in_concatenation)
    
    i = 0
    n_files = 10000
    while i + n_files <= len(ids_directory) + n_files:
        print(f"Exporting Visual GloVe embeddings")
        
        print(f"Loading files from {cfg.visual_features_dir} - Batch {i}-{i+n_files}")
        
        _, data_rgb, files_rgb, data_flow, files_flow = load_feature_data(ids_directory, directory=cfg.visual_features_dir, 
                                                                            file_rgb_posfix=cfg.file_rgb_posfix, 
                                                                            file_flow_posfix=cfg.file_flow_posfix, 
                                                                            file_extension=cfg.visual_file_extension, 
                                                                            use_flow=cfg.use_flow_in_concatenation, 
                                                                            start=i, end=i+n_files)      
        print(f"Embedding videos")
        predict_visual_embedding(glove, cluster, data_rgb, files_rgb, data_flow, files_flow, cfg.output_embedding_dir, cfg.embedding_file_extension, cfg.output_concatenated_stack_embedding, cfg.concatenated_file_extension, use_flow=cfg.use_flow_in_concatenation)
        i = i + n_files


def training_models(cfg):

    train_ids = json.loads(open(cfg.train_ids_file).read())        


    print("Loading cluster training data...")
    
    _, data, _, _, _ = load_feature_data(train_ids, directory=cfg.visual_features_dir, 
                                         file_rgb_posfix=cfg.file_rgb_posfix, 
                                         file_flow_posfix=cfg.file_flow_posfix, 
                                         file_extension=cfg.visual_file_extension, use_flow=cfg.use_flow, end=20000000)        

    data = np.concatenate(data)
    print(f"{data.shape[0]} tokens with {data.shape[1]}-d loaded")  
    print(f"Training Mini-Batch K-means...")
    cluster = mini_batch_k_means_clustering(data, output_file=cfg.c_file, 
                                            epochs=cfg.c_epochs, n_clusters=cfg.vocabulary_size, 
                                            random_state=cfg.c_random_state, 
                                            batch_size=cfg.c_batch_size, save_file=True)    


    print(f"Loading Visual GloVe training data...")
    print(f"We need to load training data again due to RAM size limitations (we have only 32 GB)")
    
    _, data, _, _, _ = load_feature_data(train_ids, directory=cfg.visual_features_dir, 
                                         file_rgb_posfix=cfg.file_rgb_posfix, 
                                         file_flow_posfix=cfg.file_flow_posfix, 
                                         file_extension=cfg.visual_file_extension, use_flow=cfg.use_flow)


    print(f"Training Visual GloVe model")
    training(cluster, data, n_epochs=cfg.vg_epochs, 
             batch_size=cfg.vg_batch_size, window_size=cfg.vg_window_size, 
             x_max=cfg.vg_x_max, alpha=cfg.vg_alpha, embed_dim=cfg.vg_emb_dim, 
             max_epochs_lower=cfg.vg_early_stopping, model_path=cfg.vg_file, 
             plot_loss=True, plot_vocabulary=True, vocab_size=cfg.vocabulary_size)


def main(args):
    print(f"Visual Glove")
    pprint(vars(args))
    start = time.time()
    cfg = Config(args)
    if cfg.procedure == "training":
        training_models(cfg)
    elif cfg.procedure == "cluster_predictions":
        cluster_predictions(cfg)
    elif cfg.procedure == "visual_glove_embeddings":
        visual_glove_embeddings(cfg)
    else:
        raise Exception("Invalid procedure.")
    print(f"Time taken: {time.time()-start} sec")
    print(f"Finish!")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visual Glove")
    parser.add_argument("--train_ids_file", type=str, default="./data/captions/train_ids.json", help="File with train ids. It is used to filter only features used to train into the visual features directory.")
    parser.add_argument("--visual_features_dir", type=str, default="./data/i3d_25fps_stack24step24_2stream_npy", help="Directory containing all features from the dataset.")
    parser.add_argument("--visual_file_extension", type=str, default=".npy", help="Input files extension (ex: .npy)")
    parser.add_argument("--file_rgb_posfix", type=str, default="_rgb", help="string identifying the rgb file (ex: _rgb)")
    parser.add_argument("--file_flow_posfix", type=str, default="_flow", help="string identifying the flow file (ex: _flow)")                
    parser.add_argument("--output_embedding_dir", type=str, default="./data/semantic_embs_128_1000_stack24_step24", help="Target directory. It is the location for the visual glove embeddings.")
    parser.add_argument("--embedding_file_extension", type=str, default=".npy", help="Output embedding file extension.")
    parser.add_argument("--output_cluster_predictions_dir", type=str, default="./data/semantic_data_1000_stack24_step24", help="Cluster target directory. It is the location for the cluster predictions.")
    parser.add_argument("--clusters_file_extension", type=str, default=".npy", help="Output clusters file extension.")    
    parser.add_argument("--output_concatenated_stack_embedding", type=str, default="./data/i3d_25fps_stack24step24_2stream_npy_semantic_embs_128_w25")
    parser.add_argument("--concatenated_file_extension", type=str, default=".npy")
    parser.add_argument("--use_flow", dest="use_flow", action="store_true", default=False, help="Whether to consider the optical flow estimation to compute the embeddings.")
    parser.add_argument("--use_flow_in_concatenation", dest="use_flow_in_concatenation", action="store_true", default=False, help="Whether to consider the optical flow estimation to compute the concatenations.")
    parser.add_argument("--c_file", type=str, default="./data/cluster.pkl", help="File containing the mini-batch k-means model.")
    parser.add_argument("--c_epochs", type=int, default=5, help="Number of epochs to training the cluster model.")
    parser.add_argument("--vocabulary_size", type=int, default=1000, help="Number of the visual words.")
    parser.add_argument("--c_batch_size", type=int, default=20000, help="Batch size used to learn mini-batch k-means")
    parser.add_argument("--device_ids", type=int, nargs="+", default=[0], help="Separated by a whitespace")    
    parser.add_argument("--vg_epochs", type=int, default=1500, help="Number of epochs used to learn visual glove.")
    parser.add_argument("--vg_batch_size", type=int, default=2048, help="Visual glove batch size")
    parser.add_argument("--vg_window_size", type=int, default=25, help="Window size used to delimit the context of a visual word in visual glove")
    parser.add_argument("--vg_x_max", type=int, default=20)
    parser.add_argument("--vg_alpha", type=int, default=0.75)
    parser.add_argument("--vg_emb_dim", type=int, default=128, help="Size of the output embeddings.")
    parser.add_argument("--vg_early_stopping", type=int, default=100, help="Inform -1 to ignore early stopping")
    parser.add_argument("--vg_file", type=str, default="./data/visualglove.pt")
    parser.add_argument('--procedure', type=str, required=True, choices=['training', 'cluster_predictions', 'visual_glove_embeddings'])
    args = parser.parse_args()
    main(args)
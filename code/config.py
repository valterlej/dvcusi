import os
import torch

class Config(object):

    def __init__(self, args):
        
        self.procedure = args.procedure
        self.train_ids_file = args.train_ids_file
        self.visual_features_dir = args.visual_features_dir
        if not os.path.isdir(self.visual_features_dir):
                raise Exception("Visual features dir do not exists.")
        self.visual_file_extension = args.visual_file_extension                
        self.file_rgb_posfix = args.file_rgb_posfix
        self.file_flow_posfix = args.file_flow_posfix
        self.output_embedding_dir = args.output_embedding_dir
        self.output_concatenated_stack_embedding = args.output_concatenated_stack_embedding
        self.concatenated_file_extension = args.concatenated_file_extension        
        self.embedding_file_extension = args.embedding_file_extension
        self.output_cluster_predictions_dir = args.output_cluster_predictions_dir
        self.clusters_file_extension = args.clusters_file_extension        
        self.use_flow = args.use_flow
        self.use_flow_in_concatenation = args.use_flow_in_concatenation
        self.c_file = args.c_file
        if self.procedure == "cluster_predictions":
            if not os.path.isfile(self.c_file):
                raise Exception("Cluster file does not exist.")
        self.c_epochs = args.c_epochs
        if self.c_epochs <= 0:
            raise Exception("Invalid number of cluster epochs training.")
        self.c_batch_size = args.c_batch_size
        if self.c_batch_size <= 1:
            raise Exception("Invalid number of cluster batch size.")
        self.c_random_state = 0
        self.vocabulary_size = args.vocabulary_size
        if self.vocabulary_size <= 1:
            raise Exception("Invalid number of vocabulary size.")
        self.device_ids = args.device_ids
        self.device = f'cuda:{self.device_ids[0]}'        
        self.vg_batch_size = args.vg_batch_size * len(self.device_ids)
        if self.vg_batch_size <= 1:
            raise Exception("Invalid visual glove batch size.")
        self.vg_epochs = args.vg_epochs
        if self.vg_epochs < 1:
            raise Exception("Invalid visual glove epochs.")
        self.vg_window_size = args.vg_window_size
        if self.vg_window_size < 2:
            raise Exception("Invalid window size context.")
        self.vg_x_max = args.vg_x_max
        if self.vg_x_max < 0 or self.vg_x_max > 100:
            raise Exception("Invalid vg_x_max value. Must be between 0 and 100.")
        self.vg_alpha = args.vg_alpha
        if self.vg_alpha < 0 or self.vg_alpha > 1:
            raise Exception("Invalid alpha. Must be between 0 and 1.")
        self.vg_emb_dim = args.vg_emb_dim
        if self.vg_emb_dim < 32:
            raise Exception("Invalid embedding dimension. We suggest values above 32.")
        self.vg_early_stopping = args.vg_early_stopping
        if self.vg_early_stopping < 0:
            self.vg_early_stopping = self.vg_epochs
        self.vg_file = args.vg_file
        if self.procedure == "visual_glove_embeddings":
            if not os.path.isfile(self.vg_file):
                raise("Invalid visual glove file model.")
        torch.cuda.set_device(self.device) 
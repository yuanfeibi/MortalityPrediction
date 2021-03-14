import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset
from functools import reduce
import os

DATA_PATH = "/Users/yuanfeibi/Documents/Tech/OMSCS/2019F/CSE6250/project/step1_featureExtraction/data/feature_csv/"
DATA_OUTPUT= "/Users/yuanfeibi/Documents/Tech/OMSCS/2019F/CSE6250/project/step1_featureExtraction/data/"
DATA_OUTPUT2= "/Users/yuanfeibi/Documents/Tech/OMSCS/2019F/CSE6250/project/step1_featureExtraction/data/processed_csv/"

features_list = ["bicarbonate", "bilirubin", "sodium", "body_temperature", "blood_pressure", "fio2", "pao2", "hr", "nitrogen", "potassium", "urine", "white_blood", "gcs"]

class VisitSequenceWithLabelDataset(Dataset):
        def __init__(self, path_list, files):
                """
                Args:
                        seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
                        labels (list): list of labels (int)
                        num_features (int): number of total features available
                """
                subject_ids = []
                labels = []
                df_patients = pd.read_csv(os.path.join(DATA_PATH, "patients/patients.csv"), usecols=["subject_id","gender","dob","dod","expire_flag"])
                df_patients['subject_id'] = df_patients['subject_id'].astype(int)
                df_patients['expire_flag'] = df_patients['expire_flag'].astype(int)

                num_features = 13
                self.seqs = []

                for i, subject_dir in enumerate(files):
                    mtx = []
                    dn = path_list[i]
                    try:
                        subject_id = int(subject_dir)
                        if not os.path.isdir(dn):
                            raise Exception
                    except:
                        continue
                    subject_ids.append(subject_id)
                    labels.append(df_patients.loc[df_patients['subject_id'] == subject_id, 'expire_flag'].iloc[0])
                    for line in open(dn + "/time_series_feature.csv"):
                        x = line.rstrip()
                        #mtx = np.zeros((len(seq), int(num_features)))
                        my_list = list(map(float, x.split(',')))
                        #my_list.append(df_patients.loc[df_patients['subject_id'] == subject_id, 'gender'].iloc[0])
                        mtx.append(my_list)
                        #print("mtx")
                        #print(type(mtx[0]))
                        #print(mtx.shape)
                    mtx2 = np.array([np.array(xi) for xi in mtx])
                    self.seqs.append(mtx2)


                self.labels = labels


        def __len__(self):
                return len(self.labels)

        def __getitem__(self, index):
                # returns will be wrapped as List of Tensor(s) by DataLoader
                return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
        """
        DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
        Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
        where N is minibatch size, seq is a (Sparse)FloatTensor, and label is a LongTensor

        :returns
                seqs (FloatTensor) - 3D of batch_size X max_length X num_features
                lengths (LongTensor) - 1D of batch_size
                labels (LongTensor) - 1D of batch_size
        """

        # TODO: Return the following two things
        # TODO: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
        # TODO: 2. Tensor contains the label of each sequence
        batch.sort(key = lambda x: len(x[0]), reverse = True)
        seqs_tensor, labels_tensor = zip(*batch)
        #print(seqs_tensor)
        #print(labels_tensor)
        batch_size = len(seqs_tensor)
        num_features = len(seqs_tensor[0][0])
        seq_len = [len(seq) for seq in seqs_tensor]
        print("num_features")
        print(num_features)
        collated_seqs = torch.zeros([batch_size, max(seq_len), num_features], dtype=torch.float32)
        #print(collated_seqs.size())
        for i, seq in enumerate(seqs_tensor):
            end_index = seq_len[i]
            '''
            print("end_index")
            print(end_index)
            print(type(end_index))
            print(num_features)
            print(type(num_features))
            print(type(collated_seqs))
            print(collated_seqs.size())
            print(i)
            print(type(i))
            print("seq")
            print(type(seq))
            print(seq[0])
            print(seq[47])
            '''
            #print(seq)
            #print(seq[:end_index, :num_features])
            collated_seqs[i, :end_index, :num_features] = torch.from_numpy(seq[:end_index, :num_features])
            #collated_seqs[i, :end_index, num_features] = seq[i, :end_index, num_features]
        return (collated_seqs, torch.FloatTensor(seq_len)), torch.LongTensor(labels_tensor)
        #return collated_seqs, torch.LongTensor(seq_len), torch.LongTensor([x.item() for x in labels_tensor])

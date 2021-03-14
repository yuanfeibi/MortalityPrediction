import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
from utils import *
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import train, evaluate
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, Dataset
from mymodels import MyVariableRNN
from mymodels import MyMLP, MyCNN, MyRNN
from plots import plot_learning_curves, plot_confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

le = preprocessing.LabelEncoder()

DATA_PATH = "/Users/yuanfeibi/Documents/Tech/OMSCS/2019F/CSE6250/project/step1_featureExtraction/data/feature_csv/"
DATA_OUTPUT= "/Users/yuanfeibi/Documents/Tech/OMSCS/2019F/CSE6250/project/step1_featureExtraction/data/"
DATA_OUTPUT2= "/Users/yuanfeibi/Documents/Tech/OMSCS/2019F/CSE6250/project/step1_featureExtraction/data/processed_csv2/"
PATH_OUTPUT = "/Users/yuanfeibi/Documents/Tech/OMSCS/2019F/CSE6250/project/model/"
features_list = ["bicarbonate", "bilirubin", "sodium", "body_temperature", "blood_pressure", "fio2", "pao2", "hr", "nitrogen", "potassium", "urine", "white_blood", "gcs"]
features_y_list = ["bicarbonate", "bilirubin", "sodium", "body_temperature", "blood_pressure", "fio2", "pao2", "hr", "nitrogen", "potassium", "urine", "white_blood", "gcs", "y"]

NUM_EPOCHS = 10
BATCH_SIZE = 32
USE_CUDA = False  # Set 'True' if you want to use GPU
NUM_WORKERS = 0
num_features = 13
device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
torch.manual_seed(1)
if device == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_data(path_list, files):
        subject_ids = []
        labels = []
        df_patients = pd.read_csv(os.path.join(DATA_PATH, "patients/patients.csv"), usecols=["subject_id","gender","dob","dod","expire_flag"])
        df_patients['subject_id'] = df_patients['subject_id'].astype(int)
        df_patients['expire_flag'] = df_patients['expire_flag'].astype(int)

        lines = []
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
            label = df_patients.loc[df_patients['subject_id'] == subject_id, 'expire_flag'].iloc[0]
            line = []
            for line in open(dn + "/time_series_feature.csv"):
                x = line.rstrip()
                line = list(map(float, x.split(',')))
                line.append(label)
            lines.append(line)
        df = pd.DataFrame(lines, columns = features_y_list)
        return df

def create_dataset(path):
        """
        :param path: path to the directory contains raw files.
        :param codemap: 3-digit ICD-9 code feature map
        :param transform: e.g. convert_icd9
        :return: List(patient IDs), List(labels), Visit sequence data as a List of List of List.
        """
        # TODO: 1. Load data from the three csv files
        # TODO: Loading the mortality file is shown as an example below. Load two other files also.
        file_list = [dI for dI in os.listdir(path) if os.path.isdir(os.path.join(path, dI))]
        #random.shuffle(file_list)
        train_files = file_list

        print("start create dataset")
        #print(train_files)
        train_path_list = []
        for subject_dir in file_list:
            dn = os.path.join(path, subject_dir)
            try:
                subject_id = int(subject_dir)
                if not os.path.isdir(dn):
                    raise Exception
            except:
                continue
            train_path_list.append(dn)

        #print(train_path_list)
        train_dataset = read_data(train_path_list, train_files)
        print("finished create dataset")
        #print(train_id)
        #print(train_label)
        return train_dataset


def main():
    # Build a code map from the train set
    '''
    df_admissions = pd.read_csv(os.path.join(DATA_PATH, "admission_type/admission_type.csv"), usecols=["subject_id", "AdmissionType"])
    df_admissions.loc[:,'Ad_M']=(df_admissions['AdmissionType']=='Medical').astype('int')
    df_admissions.loc[:,'Ad_U']=(df_admissions['AdmissionType']=='UnscheduledSurgical').astype('int')
    df_admissions.loc[:,'Ad_S']=(df_admissions['AdmissionType']=='ScheduledSurgical').astype('int')

    df_admissions = df_admissions.drop(columns=['AdmissionType'])

    #df_aids = pd.read_csv(os.path.join(DATA_PATH, "aids/aids.csv"), usecols=["subject_id", "aids"])
    df_icustays = pd.read_csv(os.path.join(DATA_PATH, "icustays/icustays.csv"), usecols=["subject_id","hadm_id","icustay_id","intime","icu_length_of_stay","age","icustay_id_order"])
    subjects = df_icustays["subject_id"].unique()

    #break_up_stays_by_subject(df_icustays, DATA_OUTPUT+"/processed_csv/", subjects=subjects)
    print("break_up_stays_by_subject done!")
    df_ages = pd.concat([df_icustays['subject_id'], df_icustays['age']], axis=1, keys=['subject_id', 'age'])
    df_ages.loc[df_ages['age'] >= 300] = 90

    df_hem = pd.read_csv(os.path.join(DATA_PATH, "hem/hem.csv"), usecols=["subject_id"])
    df_hem['hem'] = 1
    df_categorial = df_admissions.set_index('subject_id').join(df_ages.set_index('subject_id')).join(df_hem.set_index('subject_id'))
    df_categorial['age'] = df_categorial['age'].fillna(df_categorial['age'].mean())
    df_categorial['hem'] = df_categorial['hem'].fillna(0)

    # time series
    #df_intime = pd.concat([df_icustays['subject_id'], df_icustays['intime']], axis=1, keys=['subject_id', 'intime'])


    df_bicarbonate = pd.read_csv(os.path.join(DATA_PATH, "bicarbonate/bicarbonate.csv"), usecols=["subject_id", "charttime","value"])
    #events_break_up_by_subject("bicarbonate", df_bicarbonate, subjects, DATA_OUTPUT+"/processed_csv/", ["subject_id", "charttime","value"])
    extr = df_bicarbonate['value'].str.extract(r"([-+]?\d*\.\d+|\d+)", expand=False)
    global_bicarbonate_avg = pd.to_numeric(extr).mean()
    print("events_break_up_by_subject bicarbonate done!")

    df_bilirubin = pd.read_csv(os.path.join(DATA_PATH, "bilirubin/bilirubin.csv"), usecols=["subject_id", "charttime","value"])
    #events_break_up_by_subject("bilirubin", df_bilirubin, subjects, DATA_OUTPUT+"/processed_csv/", ["subject_id", "charttime","value"])
    extr = df_bilirubin['value'].str.extract(r"([-+]?\d*\.\d+|\d+)", expand=False)
    global_bilirubin_avg = pd.to_numeric(extr).mean()

    print("events_break_up_by_subject bilirubin done!")

    df_sodium = pd.read_csv(os.path.join(DATA_PATH, "sodium/sodium.csv"), usecols=["subject_id", "charttime","value"])
    #events_break_up_by_subject("sodium", df_sodium, subjects, DATA_OUTPUT+"/processed_csv/", ["subject_id", "charttime","value"])
    extr = df_sodium['value'].str.extract(r"([-+]?\d*\.\d+|\d+)", expand=False)
    global_sodium_avg = pd.to_numeric(extr).mean()
    print("events_break_up_by_subject sodium done!")

    df_body = pd.read_csv(os.path.join(DATA_PATH, "body_temperature/body_temperature.csv"), usecols=["subject_id", "charttime","value"])
    print(df_body.head())
    print(df_body.dtypes)
    #df_body.loc["value"] = df_body['value'].str.extract(r"([-+]?\d*\.\d+|\d+)", expand=False)
    df_body.loc["value"] = df_body.apply(lambda row: row['value'] if row['value'] >=50 else row['value']*1.8000 + 32.00, axis=1)
    #events_break_up_by_subject("body_temperature", df_body, subjects, DATA_OUTPUT+"/processed_csv/", ["subject_id", "charttime","value"])
    global_body_avg = df_body["value"].mean()
    print("events_break_up_by_subject body_temperature done!")

    df_blood = pd.read_csv(os.path.join(DATA_PATH, "blood_pressure/blood_pressure.csv"), usecols=["subject_id", "charttime","value"])
    #events_break_up_by_subject("blood_pressure", df_blood, subjects, DATA_OUTPUT+"/processed_csv/", ["subject_id", "charttime","value"])
    print(df_blood.dtypes)
    if df_blood['value'].dtype == str:
        extr = df_blood['value'].str.extract(r"([-+]?\d*\.\d+|\d+)", expand=False)
        global_blood_avg = pd.to_numeric(extr).mean()
    else:
        global_blood_avg = df_blood["value"].mean()
    print("events_break_up_by_subject blood_pressure done!")

    df_fio2 = pd.read_csv(os.path.join(DATA_PATH, "fio2/fio2.csv"), usecols=["subject_id", "charttime","value"])
    #events_break_up_by_subject("fio2", df_fio2, subjects, DATA_OUTPUT+"/processed_csv/", ["subject_id", "charttime","value"])
    if df_fio2['value'].dtype == str:
        extr = df_fio2['value'].str.extract(r"([-+]?\d*\.\d+|\d+)", expand=False)
        global_fio2_avg = pd.to_numeric(extr).mean()
    else:
        global_fio2_avg = df_fio2["value"].mean()
    print("events_break_up_by_subject fio2 done!")

    df_pao2 = pd.read_csv(os.path.join(DATA_PATH, "pao2/pao2.csv"), usecols=["subject_id", "charttime","value"])
    #events_break_up_by_subject("pao2", df_pao2, subjects, DATA_OUTPUT+"/processed_csv/", ["subject_id", "charttime","value"])
    extr = df_pao2['value'].str.extract(r"([-+]?\d*\.\d+|\d+)", expand=False)
    global_pao2_avg = pd.to_numeric(extr).mean()
    print("events_break_up_by_subject pao2 done!")

    df_hr = pd.read_csv(os.path.join(DATA_PATH, "hr/hr.csv"), usecols=["subject_id", "charttime","value"])
    #events_break_up_by_subject("hr", df_hr, subjects, DATA_OUTPUT+"/processed_csv/", ["subject_id", "charttime","value"])
    print(df_hr['value'].dtype)
    if df_hr['value'].dtype == str:
        extr = df_hr['value'].str.extract(r"([-+]?\d*\.\d+|\d+)", expand=False)
        global_hr_avg = pd.to_numeric(extr).mean()
    else:
        global_hr_avg = df_hr["value"].mean()
    print("events_break_up_by_subject hr done!")

    df_nitrogen = pd.read_csv(os.path.join(DATA_PATH, "nitrogen/nitrogen.csv"), usecols=["subject_id", "charttime","value"])
    #events_break_up_by_subject("nitrogen", df_nitrogen, subjects, DATA_OUTPUT+"/processed_csv/", ["subject_id", "charttime","value"])
    print(df_nitrogen['value'].dtype)
    print(df_nitrogen.head())
    if df_nitrogen['value'].dtype != str:
        df_nitrogen['value'] = df_nitrogen['value'].astype(str)
    extr = df_nitrogen['value'].str.extract(r"([-+]?\d*\.\d+|\d+)", expand=False)
    global_nitrogen_avg = pd.to_numeric(extr).mean()
    print("events_break_up_by_subject nitrogen done!")


    df_potassium = pd.read_csv(os.path.join(DATA_PATH, "potassium/potassium.csv"), usecols=["subject_id", "charttime","value"])
    #events_break_up_by_subject("potassium", df_potassium, subjects, DATA_OUTPUT+"/processed_csv/", ["subject_id", "charttime","value"])
    extr = df_potassium['value'].str.extract(r"([-+]?\d*\.\d+|\d+)", expand=False)
    global_potassium_avg = pd.to_numeric(extr).mean()
    print("events_break_up_by_subject potassium done!")


    df_urine = pd.read_csv(os.path.join(DATA_PATH, "urine/urine.csv"), usecols=["subject_id", "charttime","value"])
    #events_break_up_by_subject("urine", df_urine, subjects, DATA_OUTPUT+"/processed_csv/", ["subject_id", "charttime","value"])
    if df_urine['value'].dtype != str:
        df_urine['value'] = df_urine['value'].astype(str)
    extr = df_urine['value'].str.extract(r"([-+]?\d*\.\d+|\d+)", expand=False)
    global_urine_avg = pd.to_numeric(extr).mean()
    print("events_break_up_by_subject urine done!")

    df_white_blood = pd.read_csv(os.path.join(DATA_PATH, "white_blood/white_blood.csv"), usecols=["subject_id", "charttime","value"])
    #events_break_up_by_subject("white_blood", df_white_blood, subjects, DATA_OUTPUT+"/processed_csv/", ["subject_id", "charttime","value"])
    if df_white_blood['value'].dtype != str:
        df_white_blood['value'] = df_white_blood['value'].astype(str)
    extr = df_white_blood['value'].str.extract(r"([-+]?\d*\.\d+|\d+)", expand=False)
    global_white_blood_avg = pd.to_numeric(extr).mean()

    print("events_break_up_by_subject white_blood done!")

    # GCS
    df_gcs = pd.read_csv(os.path.join(DATA_PATH, "gcs/gcs.csv"), usecols=["icustay_id","charttime","gcs"])
    df_gcs2 = df_gcs.join(df_icustays.set_index('icustay_id'), on='icustay_id')
    df_gcs3 = pd.concat([df_gcs2['subject_id'], df_gcs2['charttime'], df_gcs2['gcs']], axis=1, keys=['subject_id', 'charttime', 'value'])
    #events_break_up_by_subject("gcs", df_gcs3, subjects, DATA_OUTPUT+"/processed_csv/", ["subject_id", "charttime","value"])
    if df_gcs3['value'].dtype != str:
        df_gcs3['value'] = df_gcs3['value'].astype(str)
    extr = df_gcs3['value'].str.extract(r"([-+]?\d*\.\d+|\d+)", expand=False)
    global_gcs_avg = pd.to_numeric(extr).mean()
    print("events_break_up_by_subject gcs done!")

    print(global_bicarbonate_avg)
    print(global_bilirubin_avg)
    print(global_sodium_avg)
    print(global_body_avg)
    print(global_blood_avg)
    print(global_fio2_avg)
    print(global_pao2_avg)
    print(global_hr_avg)
    print(global_nitrogen_avg)
    print(global_potassium_avg)
    print(global_urine_avg)
    print(global_white_blood_avg)
    print(global_gcs_avg)
    '''

    #create_dataset_by_subject(DATA_OUTPUT + "processed_csv/", [global_bicarbonate_avg, global_bilirubin_avg, global_sodium_avg, global_body_avg, global_blood_avg, \
    #                          global_fio2_avg, global_pao2_avg, global_hr_avg, global_nitrogen_avg, global_potassium_avg, global_urine_avg, global_white_blood_avg, \
    #                          global_gcs_avg])
    #df_patients = pd.read_csv(os.path.join(DATA_PATH, "patients/patients.csv"), usecols=["subject_id","gender","dob","dod","expire_flag"])
    #df_labels = pd.concat([df_patients['subject_id'], df_patients['expire_flag']], axis=1, keys=['subject_id', 'label'])

    train_dataset = create_dataset(DATA_OUTPUT2)
    print(train_dataset.shape)
    print(list(train_dataset.columns))
    print(train_dataset['y'].value_counts())
    X = train_dataset[features_list]
    y = train_dataset["y"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    print("train_dataset")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()

import os
import sys
import csv
import time
import numpy as np
import pandas as pd
import datetime
import re
import torch

features_list = ["bicarbonate", "bilirubin", "sodium", "body_temperature", "blood_pressure", "fio2", "pao2", "hr", "nitrogen", "potassium", "urine", "white_blood", "gcs"]

def break_up_stays_by_subject(stays, output_path, subjects):
    subjects = stays.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for i, subject_id in enumerate(subjects):
        dn = os.path.join(output_path, str(subject_id))
        sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i+1, nb_subjects))
        try:
            os.makedirs(dn)
        except:
            pass
        stays.ix[stays["subject_id"] == subject_id].sort_values(by='intime').to_csv(os.path.join(dn, 'stays.csv'), index=False)

def events_break_up_by_subject(event, data, subjects_to_keep, output_path, obs_header):
    for index, row in data.iterrows():
        if (subjects_to_keep is not None) and (row['subject_id'] not in subjects_to_keep):
            continue
        dn = os.path.join(output_path, str(int(row['subject_id'])))
        try:
            os.makedirs(dn)
        except:
            pass
        fn = os.path.join(dn, '{}.csv'.format(str(event)))
        if not os.path.exists(fn) or not os.path.isfile(fn):
            f = open(fn, 'w')
            f.write(','.join(obs_header) + '\n')
            f.close()
        w = csv.writer(open(fn, 'a'), delimiter =',')
        row_list = list()
        for label in obs_header:
            row_list.append(row[label])
        w.writerow(row_list)

def read_subject_intime(data_path, subject_id):
    df = pd.read_csv(os.path.join(data_path, "stays.csv"), usecols=["subject_id","intime"])
    return df.iloc[0]["intime"]

def is_valid(intime, endtime, mytime):
    if mytime >= intime and mytime <= endtime:
        return True
    return False

def create_dataset_by_subject(data_folder, global_avgs):
    for subject_dir in os.listdir(data_folder):
        dn = os.path.join(data_folder, subject_dir)
        try:
            subject_id = int(subject_dir)
            if not os.path.isdir(dn):
                raise Exception
        except:
            continue

        intime = pd.to_datetime(read_subject_intime(data_folder + str(subject_id), subject_id))
        intime_epoch = intime.timestamp()
        endtime = intime + datetime.timedelta(days=2)
        endtime_epoch = endtime.timestamp()

        # 48 hours, n = len(features_list). so 48 * n
        seqs = np.zeros((48, len(features_list)))
        for i, feature in enumerate(features_list):
            try:
                df = pd.read_csv(os.path.join(data_folder + str(subject_id) + "/", "{}.csv".format(feature)), usecols=["subject_id", "charttime", "value"])
                for index, row in df.iterrows():
                    if is_valid(intime, endtime, pd.to_datetime(row["charttime"])):
                        hour_num = (pd.to_datetime(row["charttime"]).timestamp() - intime_epoch)/3600 - 1
                        if hour_num < 0:
                            hour_num = 0
                        elif hour_num >= 48:
                            hour_num = 47
                        if type(row["value"]) != str:
                            my_str = str(row["value"])
                        else:
                            my_str = row["value"]
                        my_value = re.findall(r"([-+]?\d*\.\d+|\d+)", my_str)
                        if (len(my_value) == 0):
                            seqs[int(hour_num), i] = 0
                        else:
                            seqs[int(hour_num), i] = my_value[0]
            except IOError:
                seqs[0, i] = global_avgs[i]

        # if one colum is all zero, fill one global_avg.
        for i in range(len(seqs[0])):
            sum_of_column = sum(row[i] for row in seqs)
            if sum_of_column == 0:
                seqs[0, i] = global_avgs[i]

        # fill forward
        for i, row in enumerate(seqs):
            if i != 0:
                for j, cell in enumerate(row):
                    if seqs[i,j] == 0 and seqs[i-1, j] != 0:
                        seqs[i, j] = seqs[i-1, j]

        # fill backward
        for i, row in reversed(list(enumerate(seqs))):
            if i != len(seqs) - 1:
                for j, cell in enumerate(row):
                    if seqs[i,j] == 0 and seqs[i+1, j] != 0:
                        seqs[i, j] = seqs[i+1, j]

        fn = os.path.join(dn, 'time_series_feature.csv')
        if not os.path.exists(fn) or not os.path.isfile(fn):
            f = open(fn, 'w')
            #f.write(','.join(features_list) + '\n')
            f.close()
        w = csv.writer(open(fn, 'a'), delimiter =',')
        for row in seqs:
            w.writerow(row)

        print("create subject")
        print(subject_id)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_batch_accuracy(output, target):
    """Computes the accuracy for a batch"""
    with torch.no_grad():

        batch_size = target.size(0)
        _, pred = output.max(1)
        correct = pred.eq(target).sum()

        return correct * 100.0 / batch_size


def train(model, device, data_loader, criterion, optimizer, epoch, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        print(type(input))
        print("input type")

        print("target")
        print(type(target))
        print(target)
        if isinstance(input, tuple):
            #for e in input:
            #    print("type of e before")
            #    print(type(e))
            #input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
            input_list = [e.to(device) if type(e) == torch.Tensor else e for e in input]
            print("length of input")
            for e in input_list:
                print("e shape")
                print(e.size())
            #input = torch.stack(input_list)
            print(type(input))
            #for e in input:
            #    print("type of e after")
            #    print(type(e))
            #    print(e.size())
        else:
            input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        print("after changing tuple to tensor")
        print(type(input))
        output = model(input)
        loss = criterion(output, target)
        assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item(), target.size(0))
        accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc=accuracy))

    return losses.avg, accuracy.avg


def evaluate(model, device, data_loader, criterion, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    results = []
    # test, score
    results_score = []
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(data_loader):

            if isinstance(input, tuple):
                input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
                #input = [e.to(device, dtype=torch.float64) if type(e) == torch.Tensor else e for e in input]
                #input = torch.stack(input)
            else:
                input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss = criterion(output, target)
            print("output")
            print(output)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), target.size(0))
            accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

            y_true = target.detach().to('cpu').numpy().tolist()
            y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
            y_score = output.detach().to('cpu').max(1)[0].numpy().tolist()
            results.extend(list(zip(y_true, y_pred)))
            results_score.extend(list(zip(y_true, y_score)))
            #print("results")
            #print(results)

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(i, len(data_loader), batch_time=batch_time, loss=losses, acc=accuracy))

    return losses.avg, accuracy.avg, results, results_score

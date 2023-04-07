import pandas as pd
import shutil
import os
import re
import math
import random
datasets=['en_disease_']
seg_char='=========='
num=0
seg_list=[]
len_list=[]
seg_num=0
all_labels={}
def get_label(s):
    label = s[s.find(';') + 1:]
    label = label[:label.find(';')].replace(' ','')
    return label
def transfer(file_name):
    have=[]
    count=0
    with open(file_name, 'r', encoding='utf-8') as f:
        content_line = [line.strip() for line in f.readlines()]
    length = len(content_line)
    filter_content_line=[]
    ad_label=[]
    transfer_label=[]
    global all_labels
    label=''
    for i in range(len(content_line)):
        if seg_char not in content_line[i]:
            filter_content_line.append(content_line[i])
            if i!=len(content_line)-1 and seg_char in content_line[i+1] and label!=get_label(content_line[i+1]):
                ad_label.append(all_labels[label])
                transfer_label.append(1)
            else:
                ad_label.append(all_labels[label])
                transfer_label.append(0)
        else:
            if content_line[i].find(';')!=-1:
                label=get_label(content_line[i])
                if label not in all_labels.keys():
                    all_labels[label]=len(all_labels.keys())
            else:
                count+=1
    return filter_content_line,ad_label,transfer_label[:-1],count
def divide_folder(source_folder,out_folder):
    global seg_num
    global seg_list
    global num
    all_file = os.listdir(source_folder)
    all_count=0
    all_count_num=0
    for file in all_file:
        content, ad_label, transfer_label,count = transfer(source_folder + file)
        if count!=0:
            all_count+=1
            all_count_num+=count
        num += 1
        if len(transfer_label)!=0:
            # Following the previous work (Transformer over Pre-trained Transformer for Neural Text Segmentation with Enhanced Topic Coherence),
            # we set the max input length to 150. Otherwise, the required GPU memory is unacceptable.
            if len(ad_label)>150:
                content=content[:150]
                transfer_label = transfer_label[:149]
                ad_label = ad_label[:150]
            with open(out_folder + 'text/' + str(num) + '.txt', 'w', encoding='utf-8') as f:
                f.writelines([str(line) + '\n' for line in content])
            with open(out_folder + 'label1/' + str(num) + '.txt', 'w', encoding='utf-8') as f:
                f.writelines([str(line) + '\n' for line in transfer_label])
            with open(out_folder + 'label2/' + str(num) + '.txt', 'w', encoding='utf-8') as f:
                f.writelines([str(line) + '\n' for line in ad_label])
            len_list.append(len(content))
            seg_length = 1
            transfer_label.append(1)
            for i in range(len(transfer_label)):
                if transfer_label[i] == 1:
                    seg_list.append(seg_length)
                    seg_num += 1
                    seg_length = 1
                else:
                    seg_length += 1
        else:
            pass
    return all_count,all_count_num
def divide(dataset):
    global all_labels
    val=dataset+'validation/'
    test=dataset+'test/'
    train=dataset+'train/'
    all_count,all_count_num=divide_folder(val,'val/')
    print(val)
    print(all_count)
    print(all_count_num)
    print(all_labels)
    all_count,all_count_num=divide_folder(test, 'test/')
    print(test)
    print(all_count)
    print(all_count_num)
    print(all_labels)
    all_count,all_count_num=divide_folder(train, 'train/')
    print(train)
    print(all_count)
    print(all_count_num)
    print(all_labels)

if os.path.exists('train/'):
    pass
else :
    os.makedirs('train/text')
    os.makedirs('train/label1')
    os.makedirs('train/label2')
    os.makedirs('val/text')
    os.makedirs('val/label1')
    os.makedirs('val/label2')
    os.makedirs('test/text')
    os.makedirs('test/label1')
    os.makedirs('test/label2')
for dataset in datasets:
    divide(dataset)


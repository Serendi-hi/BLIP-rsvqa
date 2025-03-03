import os
import json
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from .utils import pre_question

from torchvision.datasets.utils import download_url

class vqa_dataset(Dataset):
    def __init__(self, transform, ann_root, vqa_root, train_files=[], split="train"):
        self.split = split
        self.transform = transform
        self.vqa_root = vqa_root

        if split=='train':
            self.annotation = []
            for f in train_files:
                self.annotation += json.load(open(os.path.join(ann_root,'%s.json'%f),'r'))
        else:
            self.annotation = json.load(open(os.path.join(ann_root,'test_qa.json'),'r'))
            self.answer_list = json.load(open(os.path.join(ann_root,'answer_list.json'),'r'))

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        if ann['dataset']=='vqa':
            image_path = os.path.join(self.vqa_root[0],ann['image'])
        elif ann['dataset']=='vg':
            image_path = os.path.join(self.vqa_root[1],ann['image'])
        elif ann['dataset']=='RSVQA-LR':
            image_path = os.path.join(self.vqa_root[2],ann['image'])
        elif ann['dataset']=='RSVQA-HR':
            image_path = os.path.join(self.vqa_root[3],ann['image'])

        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        if self.split == 'test':
            question = pre_question(ann['question'])   
            question_id = ann['question_id']
            answer=ann['answer']
            type = ann['type']
            return image, question, question_id, answer, type

        elif self.split=='val':

            question = pre_question(ann['question'])
            question_id = ann['question_id']
            type_str = ann['type']
            if ann['dataset']=='vqa':
                answer_weight = {}
                for answer in ann['answer']:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1/len(ann['answer'])
                    else:
                        answer_weight[answer] = 1/len(ann['answer'])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())
            else:
                answers = [ann['answer']]
                weights = [0.2]

            return image, question,answers, weights

        elif self.split=='train':

            question = pre_question(ann['question'])        
            
            if ann['dataset']=='vqa':               
                answer_weight = {}
                for answer in ann['answer']:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1/len(ann['answer'])
                    else:
                        answer_weight[answer] = 1/len(ann['answer'])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())

            # elif ann['dataset']=='vg':
            else:
                answers = [ann['answer']]
                weights = [0.2]  

            return image, question, answers, weights
        
        
def train_vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def val_vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n
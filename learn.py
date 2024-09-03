import os
import json
import random
import numpy as np
from PIL import Image
from models.med import BertConfig, BertModel, BertLMHeadModel
from torchvision import transforms
import torch
import torch.nn.functional as F
from models.vit import VisionTransformer, interpolate_pos_embed
from torch.utils.data import Dataset, DataLoader
from data.utils import pre_question
from models.blip import init_tokenizer
from transform.randaugment import RandomAugment
from torchvision.transforms.functional import  InterpolationMode

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(480,scale=(0.5, 1.0),interpolation=InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                       'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
    transforms.ToTensor(),
    normalize,
])

class vqa_dataset(Dataset):
    def __init__(self, ann_root, vqa_root, vg_root, transform=transform_train, split='train'):

        # self.annotation = json.load(open(os.path.join(ann_root,'vqa_train.json'),'r'))
        # self.annotation = json.load(open(os.path.join(ann_root,'vqa_val.json'),'r'))
        self.annotation = json.load(open(os.path.join(ann_root,'vg_qa.json'),'r'))
        self.answer_list = json.load(open(os.path.join(ann_root,'answer_list.json'),'r'))
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.transform = transform
        self.answer_list = json.load(open(os.path.join(ann_root,'answer_list.json'),'r'))
        self.split = split

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        if ann['dataset'] == 'vqa':
            image_path = os.path.join(self.vqa_root,ann['image'])
        elif ann['dataset'] == 'vg':
            image_path = os.path.join(self.vg_root,ann['image'].split('/')[1])

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        if self.split == 'test':
            question = pre_question(ann['question'])
            question_id = ann['question_id']
            return image, question, question_id

        elif self.split == 'train':

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

            elif ann['dataset']=='vg':
                answers = [ann['answer']]
                weights = [0.2]

            return image, question, answers, weights

def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n

# question_states = tile(question_states, 0, k_test)
def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile #(128,1,1)
    x = x.repeat(*(repeat_idx)) #question_states:(4,12,768) x:(512,12,768)
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])) #(4,128)
    return torch.index_select(x, dim, order_index.to(x.device))


if __name__ == '__main__':
    ann_root = '/home/serendi/Data/vqa_annotations'
    vqa_root = '/home/serendi/Data/mscoco'
    vg_root = '/home/serendi/Data/visual-genome'

    train_dataset = vqa_dataset(ann_root, vqa_root, vg_root,split='train')
    test_dataset = vqa_dataset(ann_root, vqa_root, vg_root,split='test')

    train_loader = DataLoader(train_dataset,
                             batch_size=4,
                             shuffle= True,
                             num_workers=4,
                             collate_fn=vqa_collate_fn)
    test_loader = DataLoader(test_dataset,
                              batch_size=4,
                              shuffle= True,
                              num_workers=4,
                              collate_fn=None)
    tokenizer = init_tokenizer()
    print('tokenizer loaded')

    cfg = BertConfig.from_json_file('configs/med_config.json')
    text_encoder = BertModel(config=cfg, add_pooling_layer=False)
    text_decoder = BertLMHeadModel(config=cfg)

    # train
    for i, (image, question, answer, weights, n) in enumerate(train_loader):
        # print('------------'+ '{}'.format(i) +'--------------')
        visual_encoder = VisionTransformer(img_size=480, patch_size=16, embed_dim=768, depth=12,
                                           num_heads=12,use_grad_checkpointing=False, ckpt_layer=0,
                                           drop_path_rate=0)
        image_embeds = visual_encoder(image)  #（4,901,768）
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long)


        ques = tokenizer(question, padding='longest', truncation=True, max_length=35, return_tensors='pt')
        ques.input_ids[:,0] = tokenizer.enc_token_id

        ans = tokenizer(answer, padding='longest', return_tensors='pt')
        ans.input_ids[:,0] = tokenizer.bos_token_id
        ans_targets = ans.input_ids.masked_fill(ans.input_ids == tokenizer.pad_token_id, -100)

        question_output = text_encoder(ques.input_ids,
                                       attention_mask=ques.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict = True) #(4,12,768)
        question_states = []
        question_atts = []
        for b,j in enumerate(n):
            question_states += [question_output.last_hidden_state[b]]*j
            question_atts += [ques.attention_mask[b]]*j
        question_states = torch.stack(question_states,0)
        question_atts = torch.stack(question_atts,0)

        answer_output = text_decoder(ans.input_ids,
                                      attention_mask = ans.attention_mask,
                                      encoder_hidden_states = question_states,
                                      encoder_attention_mask = question_atts,
                                      labels = ans_targets,
                                      return_dict = True,
                                      reduction = 'none',
                                          )

        loss = weights * answer_output.loss
        loss = loss.sum()/image.size(0)


        # rank answer
    # answer_ids = model(image, question, answer_candidates, train=False, inference='rank', k_test=128)

    # for n, (image, question, question_id) in enumerate(data_loader)):
    #     answer_ids = model(image, question, answer_candidates, train=False, inference='rank', k_test=config['k_test'])
    #
    # image_embeds = self.visual_encoder(image)
    # image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
    #
    # question = self.tokenizer(question, padding='longest', truncation=True, max_length=35,
    # return_tensors="pt").to(image.device)
    # question.input_ids[:,0] = self.tokenizer.enc_token_id
    #
    # max_ids = self.rank_answer(question_output.last_hidden_state, question.attention_mask,
    # answer.input_ids, answer.attention_mask, k_test)
#     k_test = 128
#     answer_list = test_dataset.answer_list
#     answer_candidates = tokenizer(answer_list, padding='longest', return_tensors='pt')  #(3128,8)
#     answer_candidates.input_ids[:,0] = tokenizer.bos_token_id
#
#     for n, (image, question, question_id) in enumerate(test_loader):
#         visual_encoder = VisionTransformer(img_size=480, patch_size=16, embed_dim=768, depth=12,
#                                            num_heads=12,use_grad_checkpointing=False, ckpt_layer=0,
#                                            drop_path_rate=0)
#         image_embeds = visual_encoder(image)  #（4,901,768）
#         image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long)
#
#
#         ques = tokenizer(question, padding='longest', truncation=True, max_length=35, return_tensors='pt')
#         ques.input_ids[:,0] = tokenizer.enc_token_id
#
#         question_output = text_encoder(ques.input_ids,
#                                         attention_mask = ques.attention_mask,
#                                         encoder_hidden_states = image_embeds,
#                                         encoder_attention_mask = image_atts,
#                                         return_dict = True)
#
# #        max_ids = self.rank_answer(question_output.last_hidden_state, ques.attention_mask,
# #        answer_candidates.input_ids, answer_candidates.attention_mask, k_test)
#
#         question_states = question_output.last_hidden_state #(4,12,768)
#         question_atts = ques.attention_mask   #(4,12)
#         answer_ids = answer_candidates.input_ids  #(3128,8)
#         answer_atts = answer_candidates.attention_mask  #(3128,8)
#
#         num_ques = question_states.size(0)  #4
#         start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token (4,1)
#
#         start_output = text_decoder(start_ids,
#                                      encoder_hidden_states = question_states,
#                                      encoder_attention_mask = question_atts,
#                                      return_dict = True,
#                                      reduction = 'none') #(4,1,30524)
#         logits = start_output.logits[:,0,:] # first token's logit (4,30524)
#
#         # topk_probs: top-k probability
#         # topk_ids: [num_question, k]
#         answer_first_token = answer_ids[:,1] #(3128,1) answer_list中每个词的第一个token对应的tokenizer ID
#         prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) #（4,3128） 对于每一个回答的第一个预测词，预测为answer_list的第一个token的概率
#         topk_probs, topk_ids = prob_first_token.topk(k_test,dim=1) #（4,128）
#         #topk_ids 为对于每个回答的第一个预测词，在由answer_list中每个answer的第一个token组成的vocabulary中预测概率最大的前k个答案所对应的的answer ID
#         # answer input: [num_question*k, answer_len]
#         input_ids = []
#         input_atts = []
#         for b, topk_id in enumerate(topk_ids):
#             input_ids.append(answer_ids.index_select(dim=0, index=topk_id)) #(128,8)
#             input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
#         input_ids = torch.cat(input_ids,dim=0) #(512,8)
#         input_atts = torch.cat(input_atts,dim=0)
#
#         targets_ids = input_ids.masked_fill(input_ids == tokenizer.pad_token_id, -100)
#
#         # repeat encoder's output for top-k answers
#         question_states = tile(question_states, 0, k_test) #(512,9,768)
#         question_atts = tile(question_atts, 0, k_test)
#
#         output = text_decoder(input_ids,
#                                attention_mask = input_atts,
#                                encoder_hidden_states = question_states,
#                                encoder_attention_mask = question_atts,
#                                labels = targets_ids,
#                                return_dict = True,
#                                reduction = 'none')
#
#         log_probs_sum = -output.loss
#         log_probs_sum = log_probs_sum.view(num_ques,k_test)
#
#         max_topk_ids = log_probs_sum.argmax(dim=1)
#         max_ids = topk_ids[max_topk_ids>=0,max_topk_ids]
#
#         print(max_topk_ids)
#         print(max_ids)














# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
import torch

from monai import data, transforms #data
import utils.dataset as custom_data
from glob import glob
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def datafold_read_(args, report=None, plan_raw=None):

    data_dir_list = []

    for dir in args.data_dir:
        data_dir_ = glob(dir + "**/data.nii.gz")
        data_dir_.sort()
        data_dir_list += data_dir_
    
    # load data and report
    data = {}
    for f in data_dir_list:
        
        d = {}        
        report_key = f.split('/')[-2]

        try:
            rep = report[report_key]                
            d["report"] = rep
            d["side"] = d["report"].split(' side')[0].split(' ')[-1]
            if plan_raw is not None:
                d["plan_raw"] = plan_raw[report_key]
        except:
            continue

        d['image'] = [f]
        d['label'] = f.replace('data.nii', 'label.nii')
        d['id'] = report_key

        if d["side"] not in data.keys():
            data[d["side"]] = [] 

        data[d["side"]].append(d)

    # split dataset
    tr = []
    val = []
    for side in data.keys():

        p_tr = 0.7
        p_val = 0.8
        tr += data[side][:int(p_tr*len(data[side])*args.p_data)] 
        if args.test_mode == 1:
            val += data[side][int(p_val*len(data[side])):]
        elif args.test_mode >= 2:
            val += data[side]
        else:
            val += data[side][int(p_tr*len(data[side])):int(p_val*len(data[side]))]

    print(">>>>>>> ratio: ", args.p_data)
    print(">>>>>>> trainset: ", len(tr))
    print(">>>>>>> valset: ", len(val))

    return tr, val


def get_loader(args, retriever=None):

    report = {}
    if isinstance(args.report_dir, list):
        for dir in args.report_dir:
            try:
                report_all = pd.read_excel(dir)
            except:
                report_all = pd.read_csv(dir)
            report.update(build_prompt(report_all, args))
    else:
        report_all = pd.read_excel(args.report_dir)
        report = build_prompt(report_all, args)
    datalist, val_files = datafold_read_(args, report=report)
    print(">>>>>>> context: ", args.context)
    
    # pre-tokenize
    if args.context:
        if retriever.text_encoder.llm:
            for i, datalist_orig in enumerate([datalist, val_files]):
                for j, data_i in enumerate(datalist_orig): 
                    try:
                        tok_txt_ = retriever.tokenizer.encode(data_i['report'])
                        m = nn.ConstantPad1d((0, max(0, retriever.max_length - retriever.context_length - len(tok_txt_))), 0)
                        tok_txt_ = m(torch.tensor(tok_txt_, dtype=torch.long))

                        if len(tok_txt_) > retriever.max_length - retriever.context_length:
                            tok_txt_ = tok_txt_[:retriever.max_length - retriever.context_length - 1] + [retriever.tokenizer.tokenizer.vocab_size]                    
                        
                        data_i['raw_report'] = data_i['report']
                        data_i['report'] = tok_txt_

                    except:
                        pass

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], image_only=False),
            transforms.EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            # transforms.Spacingd(
            #     keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            # ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.ScaleIntensityRanged(
                keys=["label"], a_min=0, a_max=args.c_max, b_min=0, b_max=args.c_max, clip=True, dtype=np.int8
            ),
            
            transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            # transforms.Resized(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            transforms.Rotate90d(keys=["image", "label"]),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=0.1,
                num_samples=args.sw_batch_size, #1 #4
                image_key="image",
                image_threshold=0,
            ),
            # transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            # transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            # transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            # transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], image_only=False),
            transforms.EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            # transforms.Spacingd(
            #     keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            # ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.ScaleIntensityRanged(
                keys=["label"], a_min=0, a_max=args.c_max, b_min=0, b_max=args.c_max, clip=True, dtype=np.int8
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            # transforms.Resized(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            transforms.Rotate90d(keys=["image", "label"]),
            # transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], image_only=False),
            transforms.EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            # transforms.Spacingd(keys="image", pixdim=(args.space_x, args.space_y, args.space_z), mode="bilinear"),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.ScaleIntensityRanged(
                keys=["label"], a_min=0, a_max=args.c_max, b_min=0, b_max=args.c_max, clip=True, dtype=np.int8
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            # transforms.Resized(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            transforms.Rotate90d(keys=["image", "label"]),
            # transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.use_normal_dataset:
        train_ds = custom_data.Dataset(data=datalist, transform=train_transform, report=report) #report
    else:
        train_ds = data.CacheDataset(
            data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
        )
    train_sampler = Sampler(train_ds) if args.distributed else None
    
    if args.test_mode:
        train_loader = None
        test_ds = custom_data.Dataset(data=val_files, transform=test_transform, report=report) 
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = [train_loader, test_loader]
    
    else:
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )

        val_ds = custom_data.Dataset(data=val_files, transform=val_transform, report=report) 
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader


def build_prompt(df, args):

    unit_no = list(df['Unit No'])
    unit_no_unique = np.unique(unit_no)
    report = {}

    for no in unit_no_unique:

        row = df.loc[df['Unit No'] == no]
        if unit_no.count(no) == 1:
            text_prompt = prepare_report(args, row, 0)
        else: # both
            text_prompt_list = []
            for i in range(row.ndim):
                text_prompt_list.append(prepare_report(args, row, i))
            if text_prompt_list[0] is not None:
                # print(no, text_prompt_list)
                text_prompt_list = np.unique(text_prompt_list) 
                text_prompt = '; '.join(text_prompt_list)
            else:
                text_prompt = None

        if text_prompt is not None:
            report[str(no)] = text_prompt + " <SEG>"
            
    return report


def prepare_report(args, row, i=0):

    t_stage = 'cT' + str(row['icT'].values[i]).replace('T','')
    n_stage = 'N' + str(row['icN'].values[i]).replace('N','')

    if n_stage == 'nan':
        print(row['icN'].values[i])
        n_stage == 'unknown'
    if t_stage == 'cnan':
        t_stage == 'unknown'
    t_stage = t_stage.replace(' or ', '/').replace(' ', '')
    
    if n_stage.find('2023') >= 0:
        time = n_stage.split('-')
        n_stage = 'N' + str(int(time[1])) + '/' + str(int(time[2].split(' ')[0])) 

    subsite = row['Subsite 1'].values[i]
    if 'Right' in subsite or 'right' in subsite or 'Rt' in subsite:
        orientation = 'right'
    elif 'Left' in subsite or 'left' in subsite or 'Lt' in subsite:
        orientation = 'left'
    elif 'Both' in subsite or 'both' in subsite:
        orientation = 'both'
        return None
    else:
        orientation = 'unknown' 

    remark = row['Remark'].values[i]
    if 'BCS' in remark:
        surgery = 'breast conserving surgery'
    elif ('Postop' in remark) | ('mastectomy' in remark):
        surgery = 'total mastectomy surgery'
    else:
        surgery = 'unknown type surgery'

    text_prompt = ', '.join([n_stage, t_stage, surgery, orientation + ' side'])
    print(text_prompt)

    return text_prompt

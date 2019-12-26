# -*- coding: utf-8 -*-
# @Time    : 2019/11/25 9:33
# @Author  : 2014Vee
# @Email   : 1976535998@qq.com
# @File    : json2dataset.py
# @Software: PyCharm
# !/usr/bin/python
# -*- coding: UTF-8 -*-
# !H:\Anaconda3\envs\new_labelme\python.exe
#说明

#本文档用于批量转换json为dataset，直接更改json_path即可生成四个路径pic，cv_mask，json，labelme_json
#pic放原图，cv_mask放mask
#**注意文件夹下只能由json文件，不能有其他文件，否则顺序读入会出错

import argparse
import json
import os
import os.path as osp
import base64
import warnings
from shutil import copyfile
import PIL.Image
import yaml

from labelme import utils

import cv2
import numpy as np
from skimage import img_as_ubyte


# from sys import argv

def main():


    json_file = '/home/wangb/nvdi/Mask_RCNN/train_data2'
    list = os.listdir(json_file)
    if not os.path.exists(json_file + '/' + 'pic'):
        os.makedirs(json_file + '/' + 'pic')
    if not os.path.exists(json_file + '/' + 'cv_mask'):
        os.makedirs(json_file + '/' + 'cv_mask')
    if not os.path.exists(json_file + '/' + 'labelme_json'):
        os.makedirs(json_file + '/' + 'labelme_json')
    if not os.path.exists(json_file + '/' + 'json'):
        os.makedirs(json_file + '/' + 'json')

    for i in range(0, len(list)):
        #python open()函数用于打开一个文件，有的时候list[i]不是文件
        path = os.path.join(json_file, list[i])
        if os.path.isfile(path):
            #拷贝文件函数，属于shutil库
            copyfile(path, json_file + '/json/' + list[i])
            data = json.load(open(path))
            img = utils.img_b64_to_arr(data['imageData'])
            lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])

            captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
            lbl_viz = utils.draw_label(lbl, img, captions)
            out_dir = osp.basename(list[i]).replace('.', '_')
            out_dir = osp.join(osp.dirname(list[i]), out_dir)
            #只取filename的前几个字符就行
            filename = out_dir[:-5]
            #文件路径
            out_dir = json_file + "/"+ 'labelme_json' + "/" + out_dir
            out_dir1 = json_file + "/" + 'pic'
            out_dir2 = json_file + "/" + 'cv_mask'

            if not osp.exists(out_dir):
                os.mkdir(out_dir)
            #写入原图
            PIL.Image.fromarray(img).save(osp.join(out_dir, 'img' + '.png'))
            PIL.Image.fromarray(img).save(osp.join(out_dir1, str(filename)+'.png'))
            #写入mask图
            utils.lblsave(osp.join(out_dir, 'label.png'), lbl)
            utils.lblsave(osp.join(out_dir2, str(filename)+'.png'), lbl)


        PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))

        with open(osp.join(out_dir, 'label_names'+'.txt'), 'w') as f:
            for lbl_name in lbl_names:
                f.write(lbl_name + '\n')

        warnings.warn('info.yaml is being replaced by label_names.txt')
        info = dict(label_names=lbl_names)
        with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
            yaml.dump(info, f, default_flow_style=False)

        fov = open(osp.join(out_dir, 'info' + '.yaml'), 'w')
        for key in info:
            fov.writelines(key)
            fov.write(':\n')
        for k, v in lbl_names.items():
            # 这里需要注意，按照标准的labelme生成的yaml文件的格式，建议转化前看看标准的什么样
            fov.write('-')
            fov.write(' ')
            fov.write(k)
            fov.write('\n')


        fov.close()
        print('Saved to: %s' % out_dir)


if __name__ == '__main__':
    main()

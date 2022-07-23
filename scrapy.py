import os
import time
import requests
import re

def imgdata_set(save_path,word,epoch):
    q=0     #停止爬取图片条件
    a=0     #图片名称
    while(True):
        time.sleep(1)
        url="https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word={}&pn={}&ct=&ic=0&lm=-1&width=0&height=0".format(word,q)
        #word=需要搜索的名字
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36 Edg/88.0.705.56'
        }
        response=requests.get(url,headers=headers)
        # print(response.request.headers)
        html=response.text
        # print(html)
        urls=re.findall('"objURL":"(.*?)"',html)
        # print(urls)
        for url in urls:
            print(a)    #图片的名字
            response = requests.get(url, headers=headers)
            image=response.content
            with open(os.path.join(save_path,"{}.jpg".format(a)),'wb') as f:
                f.write(image)
            a=a+1
        q=q+20
        if (q/20)>=int(epoch):
            break

import shutil
import numpy as np
from PIL import Image
import os

def 比较图片大小(dir_image1, dir_image2):
    with open(dir_image1, "rb") as f1:
        size1 = len(f1.read())
    with open(dir_image2, "rb") as f2:
        size2 = len(f2.read())
    if(size1 == size2):
        result = "大小相同"
    else:
        result = "大小不同"
    return result


def 比较图片尺寸(dir_image1, dir_image2):
    image1 = Image.open(dir_image1)
    image2 = Image.open(dir_image2)
    if(image1.size == image2.size):
        result = "尺寸相同"
    else:
        result = "尺寸不同"
    return result


if __name__ == '__main__':

    load_path = 'D:\python_dazuoye\data\\train\Zhou Jielun'  # 要去重的文件夹
    save_path = 'D:\python_dazuoye\laji'  # 空文件夹，用于存储检测到的重复的照片
    os.makedirs(save_path, exist_ok=True)

    # 获取图片列表 file_map，字典{文件路径filename : 文件大小image_size}
    file_map = {}
    image_size = 0
    # 遍历filePath下的文件、文件夹（包括子目录）
    for parent, dirnames, filenames in os.walk(load_path):
        # for dirname in dirnames:
        # print('parent is %s, dirname is %s' % (parent, dirname))
        for filename in filenames:
            # print('parent is %s, filename is %s' % (parent, filename))
            # print('the full name of the file is %s' % os.path.join(parent, filename))
            image_size = os.path.getsize(os.path.join(parent, filename))
            file_map.setdefault(os.path.join(parent, filename), image_size)

    # 获取的图片列表按 文件大小image_size 排序
    file_map = sorted(file_map.items(), key=lambda d: d[1], reverse=False)
    file_list = []
    for filename, image_size in file_map:
        file_list.append(filename)

    # 取出重复的图片
    file_repeat = []
    for currIndex, filename in enumerate(file_list):
        dir_image1 = file_list[currIndex]
        dir_image2 = file_list[currIndex + 1]

    # 将重复的图片移动到新的文件夹，实现对原文件夹降重
    for image in file_repeat:
        shutil.move(image, save_path)
        print("正在移除重复照片：", image)

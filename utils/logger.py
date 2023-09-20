'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2022-10-26 23:07:16
'''
from ast import While
from copy import deepcopy
from turtle import color
import torch
import numpy as np
from prettytable import PrettyTable
import logging
import os

NORMAL = 0
BOLD = 1
UNDERLINE = 2
INVERT = 3
BLACK= 30
RED= 31
GREEN = 32
YELLOW= 33
BLUE= 34
PURPLE= 35
CYAN= 36
WHITE = 37


def printer(func):
    def wrapper(*args, **kwargs):
        color = WHITE
        style = NORMAL
        if 'color' in kwargs.keys():
            color = kwargs['color']
        if 'style' in kwargs.keys():
            style = kwargs['style']
        print('\033[{0};{1}m'.format(style,color))
        func(*args, **kwargs)
        print('\033[1;37m')
    return wrapper

class Log():
    def __init__(self,filename,fieldname,dec=4) -> None:
        self.filename = filename
        self.dec=dec

        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s \n%(message)s\n')
        console_hander = logging.StreamHandler()
        console_hander.setLevel(logging.DEBUG)
        console_hander.setFormatter(formatter)

        file_hander = logging.FileHandler(filename,'a',encoding='UTF-8' )
        file_hander.setFormatter(formatter)
        file_hander.setLevel(logging.DEBUG)

        self.log.addHandler(console_hander)
        self.log.addHandler(file_hander)

        self.table = PrettyTable()
        self.table.field_names=fieldname
        self.header = self.table 

    @printer
    def head(self,kw,color=WHITE,style=NORMAL):
        table = PrettyTable()
        table.field_names=['para','value']
        for key,value in kw.items():
            table.add_rows([[str(key),str(value)]])
        self.log.info(table)

    @printer
    def update(self,table,color=WHITE,style=NORMAL):
        if isinstance(table,torch.Tensor):
            table = np.around(table.numpy(),self.dec)
        elif isinstance(table,np.ndarray):
            table = np.around(table,self.dec)
        
        self.table.add_rows(table)

        # print(self.table)
        # self.log.debug('1234563')
        # print("\033[34m")
        self.log.debug(self.table)

    # @printer
    def print(self,table,color=WHITE,style=NORMAL):
        if isinstance(table,torch.Tensor):
            table = np.around(table.numpy(),self.dec)
        elif isinstance(table,np.ndarray):
            table = np.around(table,self.dec)
        head = deepcopy(self.header)
        head.add_rows(table)


        # print(self.table)
        # self.log.debug('1234563')
        # print("\033[34m")
        self.log.debug(head)

if __name__=='__main__':
    # p=Printer()
    # p(123)
    l = Log('/nvme0n1/lmj/disorder_selfsup/tools/test.log',['time','epoch','learning rate','train loss','val loss','accuracy'],dec=6)
    l.head({'lr':0.001,'bs':16},color=RED)
    x = [[1,2,3,4,5,6.1234567]]
    l.print(x,color=BLUE,style=BOLD)
    x = [['6','5','4','3','2','1']]
    l.print(x,color=BLUE,style=BOLD)
'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-09-19 00:02:56
'''

import yaml
from pprint import pprint,pformat

class Config():
    def __init__(self,path='config/config.yaml') -> None:
        file = open(path,'r',encoding="utf-8")
        self.file_data = file.read()
        file.close()
        self.config = yaml.load(self.file_data,Loader=yaml.FullLoader)
        for k,v in self.config.items():
            setattr(self,k,v)
    def __call__(self):
        return self.config
    
    def __str__(self):
        return pformat(self.config)


if __name__ == '__main__':
    cfg =  Config()
    print(cfg)
# 根据SSD代码，实现TextBoxes
## ssd 代码地址：https://github.com/amdegroot/ssd.pytorch
## TextBoxes论文地址: https://arxiv.org/abs/1611.06779

## 数据集  
    ICDAR
    
## 已完成  
### train  
```
    cd Textboxes/
    sh train.sh
```
### test
``` 
    cd Textboxes/
    python3 test.py
``` 
    
### eval
```
    cd Textboxes/
    python3 eval.py   #生成gt.xml和det.xml
    evalfixed det.xml gt.xml
```
## 待完善  
    data argument  
    data transform  
        resize
        substract mean
    batch

## 待支持    
    demo  
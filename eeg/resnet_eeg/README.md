# 实验记录

## log1: 扩展数据集的实验

## log2: 不扩展数据集的实验

## log3：不扩展数据集，减少网络层数的实验

## log4: 将num_residual_units从2设置成3 

- 10000 step result
  ``` 
    INFO:tensorflow:Restoring parameters from ../log4\model.ckpt-10881
    INFO:tensorflow:loss: 0.691, precision: 0.804, best precision: 0.804
  ```
  
- 15000 step result
  ```
    INFO:tensorflow:Restoring parameters from ../log4\model.ckpt-14819
    INFO:tensorflow:loss: 0.985, precision: 0.745, best precision: 0.745
  ```

## log5：使用的同样是扩展的数据集，同时将num_residual_units从2设置成3 

- 19000 step result
  ```
    INFO:tensorflow:Restoring parameters from ../log5\model.ckpt-18876
    INFO:tensorflow:loss: 7.324, precision: 0.031, best precision: 0.033
  ```
  
## log6: 用第二个数据集，num_residual_units = 4实验

- 6000 step result
  ```
    INFO:tensorflow:Restoring parameters from ../log6\model.ckpt-5972
    INFO:tensorflow:loss: 0.693, precision: 0.814, best precision: 0.814
  ```
  
- 20000 step result
  ```
    INFO:tensorflow:Restoring parameters from ../log6\model.ckpt-21307
    INFO:tensorflow:loss: 0.933, precision: 0.843, best precision: 0.843
  ```
  
- 25000 step result
  ```
    INFO:tensorflow:Restoring parameters from ../log6\model.ckpt-25302
    INFO:tensorflow:loss: 0.949, precision: 0.848, best precision: 0.848
  ```
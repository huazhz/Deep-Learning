# 实验记录

## resnet_data1
>扩展的数据集

## resnet_data2
>1:32的数据集

## resnet_data3
>14:45的数据集

## log1: 扩展数据集的实验
>75%

## log2: 不扩展数据集的实验
>79%

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
  
- 37000 step result
  ```
    INFO:tensorflow:Restoring parameters from ../log6\model.ckpt-36792
    INFO:tensorflow:loss: 1.556, precision: 0.656, best precision: 0.656
  ```

## log7: 用第三个数据集，num_residual_units = 4实验

- 7000 step result
  ```
    INFO:tensorflow:Restoring parameters from ../log7\model.ckpt-6766
    INFO:tensorflow:loss: 0.891, precision: 0.686, best precision: 0.686
  ```
 
- 12000 step result
  ```
    INFO:tensorflow:Restoring parameters from ../log7\model.ckpt-12467
    INFO:tensorflow:loss: 0.677, precision: 0.822, best precision: 0.822
  ```
  
- 16000 step result
  ```
    INFO:tensorflow:Restoring parameters from ../log7\model.ckpt-15666
    INFO:tensorflow:loss: 0.995, precision: 0.773, best precision: 0.773
  ```

- 28000 step result
  ```
    INFO:tensorflow:Restoring parameters from ../log7\model.ckpt-27906
    INFO:tensorflow:loss: 0.871, precision: 0.748, best precision: 0.748
  ```
  
## log8：用第二个数据集，num_residual_units = 5实验
> 把batch_size从16改成了8

- 3000 step result
  ```
    INFO:tensorflow:Restoring parameters from ../log8\model.ckpt-1675
    INFO:tensorflow:loss: 1.869, precision: 0.560, best precision: 0.560
  ```
- 7000 step result
  ```
    INFO:tensorflow:Restoring parameters from ../log8\model.ckpt-6558
    INFO:tensorflow:loss: 1.482, precision: 0.641, best precision: 0.641
  ```
- 10000 step result
  ```
    INFO:tensorflow:Restoring parameters from ../log8\model.ckpt-9858
    INFO:tensorflow:loss: 2.445, precision: 0.313, best precision: 0.313
  ```
  
## log9: 用第二个数据集，num_residual_units = 5实验 batchsize=16
  
- 6000 step result
  ```
    INFO:tensorflow:Restoring parameters from ../log9\model.ckpt-5960
    INFO:tensorflow:loss: 0.576, precision: 0.853, best precision: 0.853
  ```
- 7500 step result
  ```
    INFO:tensorflow:Restoring parameters from ../log9\model.ckpt-7560
    INFO:tensorflow:loss: 0.875, precision: 0.822, best precision: 0.826
  ```
- 10000 step result
  ```
    INFO:tensorflow:Restoring parameters from ../log9\model.ckpt-10043
    INFO:tensorflow:loss: 0.508, precision: 0.844, best precision: 0.846
  ```
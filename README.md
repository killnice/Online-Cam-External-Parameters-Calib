# Online-Cam-External-Parameters-Calib
Online Cam External Parameters Calib

本项目使用python实现了在线外参估计功能，基于Bundle Adjustment算法和李群-李代数工具，我们将相机位姿估计问题解耦并线性化，使用简单的数值优化方法优化位姿
## requirement 版本和库要求
  - opencv-python <= 3.4
  - json
  - time
## install 依赖库安装
  - pip install opencv-python==3.4.1.15
  - pip install opencv-contrib-python==3.4.1.15

## run 运行
  - python calib_try.py
## 实际效果
<img src="https://user-images.githubusercontent.com/30570256/203496358-82be1448-cd69-4920-ba97-dbc6acb0b8de.png" width="600" height="400" alt="微信小程序"/><br/>

<img src="https://user-images.githubusercontent.com/30570256/203496293-291ad2b1-c073-4104-814c-2161e8c70fc7.png" width="600" height="600" alt="微信小程序"/><br/>



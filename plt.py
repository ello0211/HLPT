import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
# 示例数据
training_data_size = []

fig = plt.figure()
# rewrite
# MRR = [6.54, 6.79, 9.44, 11.81, 11.94, 12.02]
# NDCG_3 = [5.64, 5.90, 8.19, 10.32, 10.41, 10.52]
# Recall_10 = [12.77, 13.68, 17.86, 22.28, 22.55, 22.67]
# Recall_100 = [24.74, 26.89, 34.61, 42.16, 42.20, 43.16]
plt.rcParams.update({"font.size":15})
x=[]
x2=[]
# expansion
HLPT = []
LoRA = []
prefix = []
y=[]
yy=[]
# 创建折线图，指定不同颜色
xs=np.linspace(min(training_data_size),max(training_data_size),500)
xx=np.linspace(min(x),max(x),100)
xx2=np.linspace(min(x2),max(x2),45)
y7=make_interp_spline(x2,y)(xx2)
y8=make_interp_spline(x2,yy)(xx2)
plt.plot(xx2, y7, label='HLPT', color='orange', marker='o',markevery=[0,6,12,22,32,39,44], markersize=10,linestyle='-')# 10 16 22 32 42 48 54
plt.plot(xx2, y8, label='LoRA', color='blue', linestyle='-')
# plt.plot(xs, y1, label='LoRA', color='blue', linestyle='-')
# plt.plot(xs, y2, label='HLPT', color='orange', linestyle='-')
# plt.plot(xs, y3, label='prefix', color='green', linestyle='-')
# plt.plot(training_data_size, Recall_10, label='prefix', color='red', linestyle='-')
#plt.plot(training_data_size, Recall_100, label='R@100', color='green', marker='o', linestyle='-')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# 添加标题和标签
plt.title('accuracy', fontsize=20)
# plt.xlabel('training data size')
# plt.ylabel('evaluation score')

# 添加图例
# plt.legend(loc='upper left', bbox_to_anchor=(0.7, 0.9))
plt.legend()

# 显示图表
plt.show()

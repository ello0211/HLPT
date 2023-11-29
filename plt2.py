import matplotlib.pyplot as plt

categories = ["method 1", "method 2", "method 3", "method 4"]
values = [1.97, 1.80, 1.82, 1.21]
start_values = [56, 56, 56, 56]
# 定义不同柱的颜色
# colors = ['#DACEAD', '#839956', '#A7805F', '#504B08']
colors = ['lightblue', 'lightgreen', 'lightcoral', 'pink']
plt.subplots_adjust(top=0.9)
# 使用循环来创建每个柱并为其指定颜色
for i in range(len(categories)):
    plt.bar(categories[i], values[i],bottom=start_values[i], color=colors[i])
for i, value in enumerate(values):
    plt.annotate(str(value+56), (i, value+55.99), ha='center', va='bottom',fontsize=13)

# plt.xlabel('Categories')
# plt.ylabel('Values')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# 添加标题和标签
plt.title('accuracy', fontsize=20)
plt.show()
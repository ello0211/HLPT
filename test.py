import torch

# # 创建示例tensor
# tensor1 = torch.tensor([1, 3, 0, 0, -2, -3, 0, 0, 0])
#
# # 想要获取的最小值的数量
# k = 3
#
# # 使用torch.topk获取最小的k个值
# min_values, min_indices = torch.topk(tensor1, k, largest=False)
#
# # 打印结果
# print("Min Values:", min_values)
# print("Min Indices:", min_indices)

x=torch.randn(3,2)
print(x[0,:].size())
y=torch.randn(3,2)
print(y)
z=torch.where(x>y,x,y)
print(z)
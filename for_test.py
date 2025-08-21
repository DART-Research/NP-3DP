import trimesh
import os
import compas_slicer.utilities as utils
input_folder_name = 'beam1B'
DATA_PATH = os.path.join(os.path.dirname(__file__), input_folder_name)
OUTPUT_PATH = utils.get_output_directory(DATA_PATH)
OBJ_INPUT_NAME = os.path.join(DATA_PATH, 'mesh.obj')
mesh = trimesh.load(OBJ_INPUT_NAME, file_type='obj')

# 获取顶点、边和面的数量
V = len(mesh.vertices)  # 顶点数
F = len(mesh.faces)     # 面数

# 计算边数E。对于每个三角形面有3条边，但每条边被两个面共享，所以需要除以2
# 注意：这种方法假设网格是封闭的且没有边界
E = (len(mesh.edges_unique))  # 使用trimesh提供的edges_unique属性获取唯一边的数量

# 计算欧拉示性数 χ
chi = V - E + F

# 根据欧拉公式计算亏格 g
g = 1 - (chi / 2)

print(f"Mesh has {V} vertices, {E} edges, and {F} faces.")
print(f"The Euler characteristic χ is: {chi}")
print(f"The genus g of the mesh{input_folder_name} is: {g}")
1/0

# import numpy as np
# from scipy.optimize import fsolve

# # 定义非线性方程组
# def equations(vars):
#     a, b, c = vars
#     eq1 = a**(-b) + c - 0.3
#     eq2 = a**(1-b) + c - 0.025
#     eq3 = a**(1.51-b) + c
#     return [eq1, eq2, eq3]

# # 初始猜测值
# initial_guess = [0.5, 1, 0.3]

# # 使用fsolve求解
# solution = fsolve(equations, initial_guess)

# # 解出的a, b, c
# a, b, c = solution
# print(f"a = {a}, b = {b}, c = {c}")
# print(a**(0-b)+c)
# print(a**(1-b)+c)
# print(a**(1.5-b)+c)
# print (0.12120715136981101**(1+0.5505394107719388)-0.012929264762091174)
# print (0.12120715136981101**(+0.5505394107719388)-0.012929264762091174)
# print (0.12120715136981101**(1.5+0.5505394107719388)-0.012929264762091174)
# _____________________________________

# list1 = [5, 3, 8]
# list2 = [4, 7, 1]

# # 创建一个函数来合并和追踪元素来源
# def merge_and_track(list_a, list_b):
#     # 使用元组 (value, origin) 的形式来存储值及其来源
#     combined_list = [(val, 'list1') for val in list_a] + [(val, 'list2') for val in list_b]
    
#     # 对合并后的列表按照值进行排序
#     sorted_list = sorted(combined_list, key=lambda x: x[0])
    
#     # 分离排序后的值和其对应的来源
#     sorted_values = [item[0] for item in sorted_list]
#     origins = [item[1] for item in sorted_list]
    
#     return sorted_values, origins

# # 调用函数并打印结果
# sorted_values, origins = merge_and_track(list1, list2)
# print("Sorted Values:", sorted_values)
# print("Origins:", origins)
# _____________________________
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Function for the third side
# def third_side_length(theta, SF):
#     theta = theta*np.pi/180
#     return np.sqrt(2*SF**2*(1 + np.cos(theta)) - 2*SF*(1 + np.cos(theta)) + 1)


# # Generate SF and theta values
# SF_values = np.linspace(0, 1, 100)   # SF ranges from 0 to 1
# theta_values = np.linspace(0, 180, 100)  # theta ranges from 0 to 180
# SF, Theta = np.meshgrid(SF_values, theta_values)

# # Calculate third side length
# C = third_side_length(Theta, SF)

# # Plot the 3D surface
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Surface plot
# surf = ax.plot_surface(SF, Theta, C, cmap='viridis', edgecolor='k', alpha=1)

# # Labels and title
# ax.set_xlabel('SF')
# ax.set_ylabel(r'$\theta$ (radians)')
# ax.set_zlabel('Gradient Normal')
# ax.set_title('3D Plot of Gradient Normal vs SF and Theta')

# # Add color bar for reference
# fig.colorbar(surf, shrink=0.5, aspect=10)

# # Show the plot
# plt.show()


# import copy
# L1=[3,4,5]
# L2=copy.copy(L1)
# L2.append(6)
# print(L1,L2)
# L3=[L1,L2]
# L4=copy.copy(L3)
# L5=copy.deepcopy(L3)
# L3[0][0]=7
# L4[1]=[1,2,3]
# print(L3,L4,L5)


import timeit
import copy

# 创建一个测试用字典
test_dict = {str(i): i for i in range(1000)}

test_dict = list(test_dict.values())
# 测试 dict.copy()
def test_copy_method():
    return test_dict.copy()

def test_deepcopy_method():
    return copy.deepcopy(test_dict)

# 测试 for 循环逐个元素复制
def test_for_loop_method():
    copied_dict = {}
    for key, value in test_dict.items():
        copied_dict[key] = value
    return copied_dict

def test_slice_method():
    return test_dict[:]

b=test_slice_method()
test_dict[0]=7
print(b[0])
# 执行测试
print("Using dict.copy():", timeit.timeit(test_copy_method, number=10000))
# print("Using for loop:", timeit.timeit(test_for_loop_method, number=10000))
print("Using deepcopy():", timeit.timeit(test_deepcopy_method, number=10000))
print("Using slice:", timeit.timeit(test_slice_method, number=10000))
from distances import load_nested_list,save_nested_list
import os
import compas_slicer.utilities as utils
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import seaborn as sns
#from visualization import load_gradient_tangent
def main():
    input_folder_name='table_2'#'data_Y_shape'#'whole'
    DATA_PATH = os.path.join(os.path.dirname(__file__), input_folder_name)
    OUTPUT_PATH = utils.get_output_directory(DATA_PATH)
    # data=load_nested_list(file_path=OUTPUT_PATH,file_name='gradient_norm.json')
    # data=load_gradient_tangent(path=DATA_PATH,folder_name='',json_name='gradient.json')
    # data=get_data_from_multiple_folder(path=os.path.dirname(__file__),folder_names=['E1','E2','E3','E4'],json_name='gradient_norm.json')#'O1','O2','O3','O4'
    # data=load_nested_list(file_path=OUTPUT_PATH,file_name='all_field_gradient_norm_high1.josn')
    # data=load_gradient_angle_from_multiple_folder(path=os.path.dirname(__file__),folder_names=['E1','E2','E3','E4'],json_name='gradient.json',flip_folders=[True,False,False,True])#
    data=get_data_from_multiple_folder(path=os.path.dirname(__file__),
                                       folder_names=['beam1A','beam1B','beam1C','beam2A','beam2B','beam2C','beam3A','beam3B','beam3C',],json_name='layer_height_error.json',nested=True)
    #'O1','O2','O3','O4''E1','E2','E3','E4'gradient_norm.json
    #['beam1A','beam1B','beam1C','beam2A','beam2B','beam2C','beam2A','beam2B','beam2C',]'layer_height_error.json'
    # ['beam1A','beam1B','beam1A'],json_name='gradient_norm.json'
    # data=get_data_from_multiple_json(path=os.path.dirname(__file__),folder_name='beam1A',json_names=['layer_height_vertices'+str(i)+'.json' for i in range(12)])#'O1','O2','O3','O4'['gradient_norm0.json','gradient_norm1.json','gradient_norm2.json','gradient_norm3.json',
    #'gradient_norm4.json','gradient_norm5.json','gradient_norm6.json','gradient_norm7.json','gradient_norm8.json','gradient_norm9.json','gradient_norm10.json','gradient_norm11.json',]
    print(max(data),min(data),len(data))
    for li,list in enumerate(data):
        data[li]=[x for x in data[li] if x<0.3 and x>-0.3]

    if True:
        for li,list in enumerate(data):
            data[li]=[x for x in data[li] if x<0.3 and x>-0.3]
        # for li,list in enumerate(data):
        #     data[li]=[(int(x*100))*0.01 for x in data[li] ]
        data=[detect_outliers(list,300) for list in data]
        # 绘制箱线图
        boxplot = plt.boxplot(data, patch_artist=True)

        # 调整线宽
        for box in boxplot['boxes']:
            box.set(linewidth=1,facecolor='none')  # 箱体的线宽

        for whisker in boxplot['whiskers']:
            whisker.set(linewidth=1)  # 须线的线宽

        for cap in boxplot['caps']:
            cap.set(linewidth=1)  # 端盖的线宽

        for median in boxplot['medians']:
            median.set(linewidth=1)  # 中位数的线宽

        # 调整异常点的大小
        for flier in boxplot['fliers']:
            flier.set(marker='o', markersize=3,linewidth=0.5, markerfacecolor='white', markeredgecolor='black')

        # 添加标题和标签
        plt.title('Boxplot with Adjusted Line Width and Outlier Size')
        plt.xlabel('Groups')
        plt.ylabel('Values')

        # 显示图表
        plt.show()
    for i,list in enumerate(data):
        print(i,len(list))
    data1 = data[0]+data[1]+data[2]
    data2 = data[3]+data[4]+data[5]
    data3 = data[6]+data[7]+data[8]
    print(len(data1),len(data3),len(data2))

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(data1, bins=50, kde=True, color='blue', alpha=0.2, label='beam 1',edgecolor='none')
    sns.histplot(data3, bins=50, kde=True, color='green', alpha=0.2, label='beam 2',edgecolor='none')
    sns.histplot(data2, bins=50, kde=True, color='orange', alpha=0.3, label='beam 3',edgecolor='none')
    # sns.kdeplot(data1, shade=True, color="b", label="beam 1")
    # sns.kdeplot(data3, shade=True, color="orange", label="beam 2")
    # sns.kdeplot(data2, shade=True, color="g", label="beam 3")
    

    # 设置图表标题和标签
    plt.title('')
    plt.xlabel('Layer Height Error')
    plt.ylabel('Number of Points')
    plt.legend()

    # 显示图表
    plt.show()
    
    data = [1/x for x in data]
    g_avg=np.average(data)
    data = [(x-g_avg)/g_avg for x in data]
    for list in data:
        save_nested_list(file_path=OUTPUT_PATH,file_name='layer_height_error.json',data=data)

    # g_avg=math.pi/180
    # data = [x/g_avg for x in data]
    #创建直方图
    # bin_size = 15
    # min_range =-60
    # max_range = 90
    bin_size = 0.02
    min_range =-0.3
    max_range =0.4
    bins = np.arange(min_range, max_range + bin_size, bin_size)
    # bins=(1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3)

    n, bins, patches = plt.hist(data, bins=bins, alpha=0.75)

    # 设置y轴的最大值
    max_y =40000 # 你可以根据需要调整这个值
    plt.ylim(0, max_y)
    # 添加标题和轴标签
    # plt.title('gradient angle distribution')
    plt.title('layer height error distribution')
    #plt.title('gradient norm distribution')
    #plt.xlabel('value')
    plt.xlabel('error')
    plt.ylabel('number of vertices')

    # 显示网格
    plt.grid(False)

    # # 在每个直方上标注具体数据
    # for i in range(len(patches)):
    #     x = (bins[i] + bins[i+1]) / 2
    #     y = n[i]
    #     if y > max_y:
    #         # 在直方图顶部显示实际数据高度
    #         plt.text(x, max_y - 20, f'{int(y)}', ha='center', va='bottom')
    #     else:
    #         # 在直方图内部显示数据高度
    #         plt.text(x, y, f'{int(y)}', ha='center', va='bottom')

    # 显示图表
    plt.show()   

def get_data_from_multiple_folder(path,folder_names,json_name,nested=False):
    datas=[]
    for folder_name in folder_names:
         data = load_json_file(path, folder_name, json_name)
         print(data)
         if nested:
             datas.append(data)
         else:
            datas+=data
    return datas

def get_data_from_multiple_json(path,folder_name,json_names):
    datas=[]
    for json_name in json_names:
         print(json_name)
         data = load_json_file(path, folder_name, json_name)
         print(data)
         datas+=data
    return datas

def load_gradient_tangent(path,folder_name,json_name):
    """
    not finished yet
    """

    data = load_json_file(path, folder_name, json_name)
    gradient_tangents = []
    
    if data:
        data_keys=sorted([int(key) for key in data.keys()])
        for vkey in data_keys:
            #print(vkey)
            gradient=data[str(vkey)]
            #print(gradient)
            gradient_tangent = math.atan(gradient[2]/(gradient[0]**2+gradient[1]**2)**0.5)
            #print(gradient_tangent)    
            gradient_tangents.append(gradient_tangent)
    return gradient_tangents

def load_gradient_angle_from_multiple_folder(path,folder_names,json_name,flip_folders):
    datas=[]
    for folder_name,flip_folder in zip(folder_names,flip_folders):
         data = load_gradient_tangent(path, folder_name, json_name)
         data=[-x for x in data] if flip_folder else data
         datas+=data
    return datas
    
def load_json_file(path, folder_name, json_name, in_output_folder=True):
    """ Loads data from json. """
    print('loading json file',folder_name,json_name)
    if in_output_folder:
        filename = os.path.join(os.path.join(path), folder_name, 'output', json_name)
    else:
        filename = os.path.join(os.path.join(path), folder_name, json_name)
    data = None

    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        print("Loaded Json: '" + filename + "'")
    else:
        print("Attention! Filename: '" + filename + "' does not exist. ")

    return data
def detect_outliers(lst,n):
    Q1 = np.percentile(lst, 25)  # 第一四分位数
    Q3 = np.percentile(lst, 75)  # 第三四分位数
    IQR = Q3 - Q1  # 四分位距
    lower_bound = Q1 - 1.5 * IQR  # 下界
    upper_bound = Q3 + 1.5 * IQR  # 上界
    outliers = [x for x in lst if x < lower_bound or x > upper_bound]  # 识别异常值
    inliers = [x for x in lst if x >= lower_bound and x <= upper_bound]
    new_outliers=[]
    outliers.sort()
    lenth=len(outliers)
    endmod=lenth%n
    for i,outlier in enumerate(outliers):
        if i%n==endmod or i%n==0 :
            new_outliers.append(outlier)
    


    return new_outliers+inliers
if __name__ == "__main__":
    main()
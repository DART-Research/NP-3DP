
from compas.datastructures import Mesh
import compas_slicer.utilities as utils
#from interpolationdart import DartPreprocesssor
from scipy.optimize import minimize
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
# import copy
import math
from typing import Set,List,Dict,Tuple
from gradient_descent import kill_local_critical
from collections import deque

class GradientOptimization:
    def __init__(self,processor,outputpath):
        self.mesh=processor.mesh
        self.target_Low=processor.target_LOW
        self.target_High=processor.target_HIGH
        self.VN=self.target_High.VN
        self.known_field={}
        self.unknown_field={}
        self.get_known_field()
        self.outputpath=outputpath
        self.processor=processor
    def get_known_field(self):
        unknown_id=0
        for vkey in self.mesh.vertices():
            if any(vkey in clustered_vkey for clustered_vkey in self.target_Low.clustered_vkeys):
                self.known_field[vkey]=0
            elif any(vkey in clustered_vkey for clustered_vkey in self.target_High.clustered_vkeys):
                self.known_field[vkey]=1
            else:
                self.unknown_field[vkey]=unknown_id
                unknown_id+=1
        self.unknown_VN=unknown_id
    
    def optimize_gradient(self,way='heat_accumulation'):
        time_o=time.time()
        initial_guess = [0.5]*self.unknown_VN
        if way == 'smooth':
            scalar_field=self.laplacian_smoothing(initial_guess,iterations=20000)
        elif way =='minimize':
            result = minimize(
                                self.objective,
                                initial_guess,
                                args=(self.mesh),
                                method='L-BFGS-B'
                            )
            
            scalar_field =union_field(result.x,self.known_field,self.unknown_field,self.mesh)
        elif way == 'combine':
            scalar_field=self.laplacian_smoothing(initial_guess,iterations=5000)
            initial_guess=split_field(scalar_field,self.unknown_field,self.mesh)
            result = minimize(
                                self.objective,
                                initial_guess,
                                args=(self.mesh),
                                method='BFGS'
                            )
            scalar_field =union_field(result.x,self.known_field,self.unknown_field,self.mesh)
        elif way == 'gradient_4D':
            scalar_field=self.laplacian_smoothing(initial_guess,iterations=1000)
            initial_guess=split_field(scalar_field,self.unknown_field,self.mesh)
            scalar_field=self.laplacian_smoothing(initial_guess,iterations=1000,way='gradient_4D')
        elif way == 'gradient_3D':
            scalar_field=self.laplacian_smoothing(initial_guess,iterations=5001)
            initial_guess=split_field(scalar_field,self.unknown_field,self.mesh)
            scalar_field=self.laplacian_smoothing(initial_guess,iterations=100,way='gradient_3D')
        elif way == 'global':
            scalar_field=global_smoothing(self.mesh,self.known_field)
        elif way == 'heat_accumulation':
            try :
                scalar_field=utils.load_from_json(filepath=self.outputpath,name='heat_accumulation_tsf.json')
            except:
                scalar_field=self.laplacian_smoothing(initial_guess,iterations=3001)
                initial_guess=split_field(scalar_field,self.unknown_field,self.mesh)
                scalar_field=self.laplacian_smoothing(initial_guess,iterations=50,way='gradient_3D')
                utils.save_to_json(filepath=self.outputpath,name='heat_accumulation_tsf.json',data=scalar_field)
            heat_accumulater_=heat_accumulater(self.mesh,scalar_field,self.processor,0.000000001)
            
            heat_accumulater_.accumulate_heat(known_values=self.known_field,unkown_values=self.unknown_field)
            # initial_guess=split_field(scalar_field,self.unknown_field,self.mesh)
            # scalar_field=self.laplacian_smoothing(initial_guess,iterations=20,way='gradient_3D')
        # print(scalar_field)
        # print(len(scalar_field))
        # print(len(union_field(scalar_field,self.known_field,self.unknown_field,self.mesh)))   
        gradients=get_face_gradient_from_scalar_field(self.mesh,scalar_field,True)
        gradients=np.array(gradients)
        gradients_norm=np.linalg.norm(gradients,axis=1)
        print('time for optimize gradient:',time.time()-time_o)
        print('vars:',np.var(gradients_norm),len(gradients_norm),'max_field:',max(scalar_field),'min_field',min(scalar_field))
        return scalar_field
    
    def objective(self, x, mesh:Mesh):
        scalar_field = union_field(x,self.known_field,self.unknown_field,self.mesh)
        gradients=get_face_gradient_from_scalar_field(mesh,scalar_field,True)
        gradients=np.array(gradients)
        gradients_norm=np.linalg.norm(gradients,axis=1)
        print('min:',min(gradients_norm))
        return( -min(gradients_norm))

    def laplacian_smoothing(self, x, iterations=10, patience=5,way='average'):
        scalar_field = union_field(x,self.known_field,self.unknown_field,self.mesh)
        min_gradient_history = []
        variance_history = []
        time_s=time.time()
        for i in range(iterations):
            new_scalar_field = scalar_field.copy()
            for vkey in self.mesh.vertices():
                if vkey in self.known_field:
                    continue  # 跳过已知点
                if way=='average':
                    neighbors = self.mesh.vertex_neighbors(vkey)
                    new_scalar_field[vkey] = sum(scalar_field[n] for n in neighbors) / len(neighbors)
                elif way=='gradient_4D':
                    neighbors = self.mesh.vertex_neighbors(vkey)
                    neighbors_coor=np.array([self.mesh.vertex_coordinates(n) for n in neighbors])
                    neighbors_scalar_field=np.array([scalar_field[n] for n in neighbors])
                    A = np.hstack([neighbors_coor[:, :3], np.ones((neighbors_coor.shape[0], 1))])
                    params = np.linalg.lstsq(A, neighbors_scalar_field, rcond=None)[0]  # params = [a1, a2, a3, b]
                    a1, a2, a3, b = params
                    vkey_coor = np.array(self.mesh.vertex_coordinates(vkey))
                    def scalar_field_func(x,params):
                        a1, a2, a3, b = params
                        return a1 * x[0] + a2 * x[1] + a3 * x[2] + b
                    new_scalar_field[vkey] =scalar_field_func(vkey_coor,params)
                    if new_scalar_field[vkey]>1 or new_scalar_field[vkey]<0:
                        print(i,vkey,'gradient:',A,neighbors_scalar_field,vkey,a1,a2,a3,b,vkey_coor,new_scalar_field[vkey],
                              [scalar_field_func(coor,params) for coor in neighbors_coor])

                        raise Exception('gradient error')
                elif way=='gradient_3D':
                   
                    smooth_vertex_gra_3D(mesh=self.mesh,vkey=vkey,scalar_field=scalar_field,new_scalar_field=new_scalar_field)
                    if new_scalar_field[vkey]>1 or new_scalar_field[vkey]<0:
                        print(i,vkey)

                        raise Exception('gradient error')
                

                

            scalar_field = new_scalar_field
            if i % 1000 == 0 or i==iterations-1:

                gradients = get_face_gradient_from_scalar_field(self.mesh, scalar_field, True)
                gradients = np.array(gradients)
                gradients_norm = np.linalg.norm(gradients, axis=1)
                time_n=time.time()
                var = np.var(gradients_norm)
                min_g = min(gradients_norm)

                print(f'Iteration {i}: vars:{var}, len_gradients:{len(gradients_norm)}, max_g:{max(gradients_norm)}, min_g:{min_g}','time:',time_n-time_s)
                time_s=time_n

                # 更新历史记录
                min_gradient_history.append(min_g)
                variance_history.append(var)

                # 当历史记录长度超过耐心值时，移除最早的记录
                if len(min_gradient_history) > patience:
                    min_gradient_history.pop(0)
                    variance_history.pop(0)

                # 检查停止条件
                if len(min_gradient_history) == patience and \
                all(min_gradient_history[i] < min_gradient_history[-1] for i in range(len(min_gradient_history)-1)) and \
                all(variance_history[i] > variance_history[-1] for i in range(len(variance_history)-1)):
                    print("Stopping early due to convergence.")
                    break
                else:
                    # 如果不符合条件，重置历史记录
                    min_gradient_history.clear()
                    variance_history.clear()            

        return scalar_field
def union_field(x,known_field,unknown_field,mesh:Mesh):
    scalar_field=[]
    for vkey in mesh.vertices():
        if vkey in known_field:
            scalar_field.append(known_field[vkey])
        elif vkey in unknown_field:                
            scalar_field.append(x[unknown_field[vkey]])
    return scalar_field
def split_field(field,unknown_field,mesh:Mesh):
    unkown_guess=[]
    for vkey in mesh.vertices():
        if vkey in unknown_field:
            unkown_guess.append(field[vkey])
    return unkown_guess
def fit_plane(points):
    """
    拟合通过给定点集的平面。
    
    :param points: N x 3的数组，每个点为 [x, y, z]
    :return: 平面系数 [A, B, C, D]，即 Ax + By + Cz + D = 0
    """
    # 构建设计矩阵（A）
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    B = points[:, 2]
  

    # 使用最小二乘法求解Ax=B，得到平面参数[x, y, 1] * [A, B, C]^T = -D
    result = np.linalg.lstsq(A, B, rcond=None)
    coeff, residuals, rank, s=result
    a,b,c = coeff
    A=-a
    B=-b
    C=1
    D=-c
    # coeff, residuals, rank, s =result
    # # 计算D
    # D = -np.mean(A.dot(coeff) - B)

    return np.array([A, B, C, D])
def fit_w_gradient(points, w_values):
    """
    拟合w关于x和y的线性关系，以计算w的梯度。
    
    :param points: N x 2的数组，每个点为 [x, y]（假设已经投影到平面上）
    :param w_values: 对应于每个点的w值
    :return: w的梯度系数 [wx, wy, b]
    """
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    
    # 使用最小二乘法求解Aw=b，得到w关于x和y的线性系数
    coefficients, residuals, rank, s = np.linalg.lstsq(A, w_values, rcond=None)
    
    return coefficients
def project_point_to_plane(point, plane_coefficients):
    """
    将一个点投影到给定的平面上。
    
    :param point: 面外点 [x1, y1, z1]
    :param plane_coefficients: 平面系数 [A, B, C, D]，即 Ax + By + Cz + D = 0
    
    :return: 投影点 [x', y', z']
    """
    A, B, C, D = plane_coefficients
    x1, y1, z1 = point
    
    # 计算比例因子t
    denominator = A**2 + B**2 + C**2
    t = (A*x1 + B*y1 + C*z1 + D) / denominator
    
    # 计算投影点坐标
    x_proj = x1 - t * A
    y_proj = y1 - t * B
    z_proj = z1 - t * C
    
    return np.array([x_proj, y_proj, z_proj])

def calculate_w_value_at_projection(plane_coefficients, w_gradient, point):
    """
    计算面外点在其平面上投影点的w值。
    
    :param plane_coefficients: 平面系数 [A, B, C, D]
    :param w_gradient: w的梯度 [wx, wy, b]
    :param point: 面外点 [x1, y1, z1]
    
    :return: 投影点的w值
    """
    # 投影点坐标
    proj_point_3d = project_point_to_plane(point, plane_coefficients)
    
    # 注意：我们只需要xy坐标来计算w值
    proj_point_2d = proj_point_3d[:2]
    
    # 使用线性方程计算投影点的w值
    wx, wy, b = w_gradient
    w_value = wx * proj_point_2d[0] + wy * proj_point_2d[1] + b
    
    return w_value

def predict_w_value(points, w_values, new_point):
    """
    预测新点在其平面上投影点的w值。
    
    :param points: N x 3的数组，每个点为 [x, y, z]
    :param w_values: 对应于每个点的w值
    :param new_point: 新点 [x_new, y_new, z_new]
    
    :return: 新点在其平面上投影点的w值
    """
    # 拟合平面
    plane_coefficients = fit_plane(points)
    # print('coefficient',plane_coefficients)
    # 将所有点投影到平面上
    projected_points = project_points_to_plane(points, plane_coefficients)
    
    # 提取投影后的xy坐标用于拟合w的梯度
    xy_projected_points = projected_points[:, :2]
    
    # 拟合w的梯度
    w_gradient = fit_w_gradient(xy_projected_points, w_values)
    
    # 计算新点在其平面上投影点的w值
    w_value_projection = calculate_w_value_at_projection(plane_coefficients, w_gradient, new_point)
    if w_value_projection>1 or w_value_projection<0:
                        print('input',points,projected_points,'plane',plane_coefficients,'w_gradient',w_gradient,
                              'ouput',new_point,w_value_projection,'test',
                              [calculate_w_value_at_projection(plane_coefficients, w_gradient, coor) for coor in points],
                               w_values)
                       

                      

    
    return w_value_projection

def project_points_to_plane(points, plane_coefficients):
    """
    将一系列点投影到给定的平面上。
    
    :param points: N x 3的数组，每个点为 [x, y, z]
    :param plane_coefficients: 平面系数 [A, B, C, D]，即 Ax + By + Cz + D = 0
    
    :return: 投影点集，N x 3的数组
    plane_coefficeients:z=ax+by+c
    """
    A,B,C,D = plane_coefficients

    projected_points = []

    for point in points:
        x1, y1, z1 = point
        
        # 计算比例因子t
        denominator = A**2 + B**2 + C**2
        t = (A*x1 + B*y1 + C*z1 + D) / denominator
        
        # 计算投影点坐标
        x_proj = x1 - t * A
        y_proj = y1 - t * B
        z_proj = z1 - t * C
        
        projected_points.append([x_proj, y_proj, z_proj])
    
    return np.array(projected_points)
def find_plane_equation(points):
    """
    根据给定的三个点计算空间中的平面方程。
    
    参数:
        points (list of lists): 一个包含三个点的列表，每个点是一个长度为3的列表或元组，表示(x, y, z)坐标。
        
    返回:
        tuple: 平面方程的系数 (a, b, c, d)，其中 a*x + b*y + c*z + d = 0。
    """
    if len(points) != 3 or not all(len(point) == 3 for point in points):
        raise ValueError("需要恰好三个点，每个点必须有三个坐标值。")
    
    # 提取点坐标
    A, B, C = points
    x1, y1, z1 = A
    x2, y2, z2 = B
    x3, y3, z3 = C
    
    # 计算向量AB和AC
    AB = [x2 - x1, y2 - y1, z2 - z1]
    AC = [x3 - x1, y3 - y1, z3 - z1]
    
    # 计算法向量n = AB × AC
    n = np.cross(AB, AC)
    
    # 确保法向量不是零向量（即三点共线的情况）
    if np.allclose(n, [0, 0, 0]):
        raise ValueError("三点共线，无法确定唯一平面。")
    
    a, b, c = n
    # 使用点A来计算d
    d = -(a * x1 + b * y1 + c * z1)
    
    return a, b, c, d

def global_smoothing(mesh:Mesh, known_values:dict, weight=1.0):
    vertices = list(mesh.vertices())
    vkey_to_index = {vkey: i for i, vkey in enumerate(vertices)}
    n = len(vertices)
    
    # 构建稀疏矩阵
    rows, cols, data = [], [], []
    b = np.zeros(n)
    
    for u, v in mesh.edges():
        idx_u = vkey_to_index[u]
        idx_v = vkey_to_index[v]
        rows += [idx_u, idx_v]
        cols += [idx_v, idx_u]
        data += [weight, weight]
    
    for vkey, value in known_values.items():
        idx = vkey_to_index[vkey]
        rows.append(idx)
        cols.append(idx)
        data.append(1e10)  # 高权重固定已知值
        b[idx] = value * 1e10
    
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    
    # 求解线性系统
    x = spsolve(A, b)
    
    # 更新场值
    result = {vkey: x[vkey_to_index[vkey]] for vkey in vertices}
    return result.values()

def smooth_vertex_gra_3D(mesh:Mesh,vkey,scalar_field,new_scalar_field,limitation=None,normalized = True):

    neighbors = mesh.vertex_neighbors(vkey)
    neighbors_coor=np.array([mesh.vertex_coordinates(n) for n in neighbors])
    vkey_coor = np.array(mesh.vertex_coordinates(vkey))
    neighbors_scalar_field=np.array([scalar_field[n] for n in neighbors])
    predict_value=predict_w_value(neighbors_coor,neighbors_scalar_field,vkey_coor)
    if (predict_value>1 or predict_value<0) and normalized:
        print(vkey)
        raise Exception('predict_value error')
    if limitation is None:
        new_scalar_field[vkey] = predict_value
    elif limitation == True:
        if predict_value<scalar_field[vkey]:
            new_scalar_field[vkey]=scalar_field[vkey]
    elif limitation == False:
        if predict_value>scalar_field[vkey]:
            new_scalar_field[vkey]=scalar_field[vkey]


def calculate_gradient_on_triangle(points, scalar_values):
    """
    计算通过给定三个非共线点及其相应标量值定义的标量场的梯度。
    
    参数:
    points : list of tuples/lists
        包含三个非共线点坐标的列表，每个元素是一个长度为3的元组或列表 [x, y, z]。
    scalar_values : list of floats
        与points中每个点对应的标量值的列表。
        
    返回:
    gradient : tuple
        梯度向量 (G_x, G_y, G_z)。
    """
    # 构建增广矩阵A，其中最后一列是1，用于表示常数项D
    if len(points) != len(scalar_values) or len(points) != 3:
        raise ValueError("需要三个点来确定平面和梯度")
    
    # 构建增广矩阵A，其中最后一列是1，用于表示常数项D
    A = np.array([[x, y, z, 1] for x, y, z in points])
    
    # 构建右侧向量b
    b = np.array(scalar_values)
    
    # 计算面ABC的法向量n
    v1 = np.array(points[1]) - np.array(points[0])
    v2 = np.array(points[2]) - np.array(points[0])
    n = np.cross(v1, v2)
    N = np.array([n[0], n[1], n[2], 0])
    
    # 添加法向量约束到A和b
    #print('3piontgra',A,N)
    A = np.vstack([A, N])
    b = np.append(b, 0)
    
    try:
        # 使用numpy求解线性方程组
        solution = np.linalg.solve(A, b)
        gradient = solution[:3]  # 只取前三个元素作为梯度
        return tuple(gradient)
    except np.linalg.LinAlgError:
        A = np.array([[x, y, z, 0] for x, y, z in points])
        A = np.vstack([A, N])
        print(points,scalar_values)
        try:
            # 使用numpy求解线性方程组
            solution = np.linalg.solve(A, b)
            gradient = solution[:3]  # 只取前三个元素作为梯度
            return tuple(gradient)
        except np.linalg.LinAlgError:
            print(points,scalar_values)
            raise ValueError("无法求解线性方程组，可能是由于点共线或其他原因")

class heat_accumulater:
    def __init__(self,mesh:Mesh,scalar_field,processor=None,min_dist=0.00000001):
        self.mesh=mesh
        self.scalar_field=scalar_field
        self.min_dist=min_dist
        self.up_down_edge={}
        self.gradients_from_low={}
        self.gradients_from_up={}
        self.processor=processor
        if processor is not None:
            
            self.get_cluster_neighbor()
        else:
            self.down_cluster_neighbors=[]
            self.up_cluster_neighbors=[]
        
    
    def get_cluster_neighbor(self):
        self.down_cluster_neighbors=[ get_neighbors(self.mesh,vkeys,round=1) for vkeys in self.processor.target_LOW.clustered_vkeys]
        self.up_cluster_neighbors=[ get_neighbors(self.mesh,vkeys,round=1) for vkeys in self.processor.target_HIGH.clustered_vkeys]
     

    def accumulate_heat(self,known_values:dict,unkown_values:dict):
        """
        accumulate_heat_with_min(mesh,known_values,unkown_values,min_gra=0)
        
        Parameters:
        mesh: Mesh object
        known_values: dict, keys are vertex keys and values are corresponding heat values
        unkown_values: dict, keys are vertex keys and values are corresponding heat values
        min_gra: float, minimum gradient value allowed
        """
        self.ve_relationship=vertex_edge_relationship(self.mesh,known_values)
        for vkey in unkown_values.keys():
        
            self.find_veky_gradient(vkey)

        self.known_values=known_values
        self.unkown_values=unkown_values
        
        
        max_round=1
        max_round_i=200
        time_oo=time.time()
        for round_id in range(max_round):
            

            time_o=time.time()
            g1=self.find_gra_min(direction=True,max_round_j=6,max_round_i=max_round_i)
            g2=self.find_gra_min(direction=False,max_round_j=6,max_round_i=max_round_i)
            g3=self.find_gra_min(direction=True,max_round_j=7,max_round_i=max_round_i)
            print('heat_ac_round_____________',round_id,time.time()-time_o,g1,g2,g3,'_______________________________________')
        print('heat_ac_total_____________',time.time()-time_oo,'_______________________________________')
    def find_gra_min(self,direction,max_round_j=3,max_round_i=100):
        # gradients = get_face_gradient_from_scalar_field(self.mesh, self.scalar_field, True)
        # gradients = np.array(gradients)
        # gradients_norm = np.linalg.norm(gradients, axis=1)
        # gradients_min=min(gradients_norm)
        # gradients_avg=np.average(gradients_norm)
        self.ve_relationship.save_back_up()
        gradients=[]
        if direction:
            for vkey in self.ve_relationship.vertex_influenced_from_low:
                self.find_veky_gradient(vkey)
            gradients=[gra[2] for gra in self.gradients_from_low.values()]
                # if max(gra_vs)<0:
                #     print('gra_max:',max(gra_vs),tuples,vkey)
        else:
            for vkey in self.ve_relationship.vertex_influenced_from_up:
                self.find_veky_gradient(vkey)
            gradients=[gra[2] for gra in self.gradients_from_up.values()]
            
        gradients = np.array(gradients)
        gradients_min=min(gradients)
        gradients_avg=np.average(gradients)
        print('gradients_min:',gradients_min,' avg:',gradients_avg,' direction:',direction)
        offset_on_min=(gradients_avg-gradients_min)
        self.gra_min=gradients_avg
        self.scalar_field_org=self.scalar_field.copy()
        for round_jd in range(max_round_j):

            offset_on_min/=2
            touch_boundary=self.accumulate_heat_one_direction(direction=direction,max_round=max_round_i)
            print('acuumulate heat round:',round_jd,'gra_min:',self.gra_min,'touch_boundary:',touch_boundary)
            if touch_boundary==True:
                #self.scalar_field=copy.deepcopy(self.scalar_field_org)
                self.gra_min-=offset_on_min
            elif touch_boundary==False:
                self.gra_min+=offset_on_min
            elif touch_boundary is None:
                break
        return self.gra_min
    
    def accumulate_heat_one_direction(self,direction=True,max_round=100):
        vertex_need=list(self.unkown_values.keys()) 
        touch_edge_all=False
        for round_id in range(max_round):
              
            touch_edge,vertex_need,edited_vkey=self.accumulate_one_round(vertex_need,direction)
            if touch_edge:
                touch_edge_all=True
            if len(vertex_need)==0:
                break
            elif round_id==max_round-1:
                print('didnot reach best round:',round_id,'vertex checked',len(vertex_need),'vertex_will_check_next:',len(vertex_need),'direction:',direction)
                touch_edge=None
            # else:
            #     print('round:',round_id,'edited_vkey',len(edited_vkey),'vertex_will_check_next:',len(vertex_need),'direction:',direction,'touch_boundary:',len(self.edited_vkey_touch_boundary))
        return touch_edge_all

     
    def accumulate_one_round(self,vertex_need:List,direction=True):
        vertex_need.sort(key=lambda x: self.scalar_field[x])
        self.edited_vkey_touch_boundary=set()
        touch_boundary_all=False
        if not direction:
            vertex_need.reverse()  
        edited_vkey=set()
        
        vkey_area=[]
        for need_id,vkey in enumerate(vertex_need):
            
            if vkey not in edited_vkey and vkey not in self.edited_vkey_touch_boundary:
                touch_boundary,new_edited_vkeys=self.accumulate_one_vertex_area(vkey,direction)
                if touch_boundary:
                    #print('vkey_area:',vkey_area)
                    #print('touch_boundary_round_one_round,loop_id:',need_id,'vkey',vkey,self.scalar_field[vkey],len(new_edited_vkeys),'areas',len(vkey_area))
                    touch_boundary_all=True
                    self.edited_vkey_touch_boundary.update(new_edited_vkeys)
                elif new_edited_vkeys:
                    vkey_area.append(len(new_edited_vkeys))
                    #print('update_scalar_field_org_for_vetex:',vkey)
                    for vkey_ in new_edited_vkeys:
                       
                        self.scalar_field_org[vkey_]= (self.scalar_field[vkey_])
                    edited_vkey.update(new_edited_vkeys)
                    # touch_boundary = False
        edited_vkey.difference_update(self.edited_vkey_touch_boundary)
        edited_vkey=list(edited_vkey)

        
        vertex_need_o=edited_vkey.copy()
        neibor_vertex=(get_neighbors(self.mesh,vertex_need_o,self.known_values,round=1,unique=True))
        for vertex in neibor_vertex:
            smooth_vertex_gra_3D(self.mesh,vertex,self.scalar_field,self.scalar_field,direction)
        neibor_vertex.update(vertex_need_o)
        for smooth_i in range(10):
            for vertex in neibor_vertex:
                smooth_vertex_gra_3D(self.mesh,vertex,self.scalar_field,self.scalar_field,direction)
        # for smooth_i in range(1):
        #     for vertex in neibor_vertex:
        #         smooth_vertex_gra_3D(self.mesh,vertex,self.scalar_field,self.scalar_field)        
        vertex_need_nei=get_neighbors(self.mesh,neibor_vertex,self.known_values,round=1,unique=False)
        vertex_need_nei.difference_update(self.edited_vkey_touch_boundary)  
        print('vertex checked',len(vertex_need_o),'vertex_smooth',len(neibor_vertex),'vertex_will_check_next:',len(vertex_need_nei),'direction:',direction,'touch_edge',touch_boundary_all,len(self.edited_vkey_touch_boundary),'areas',len(vkey_area))

        return touch_boundary_all,list(vertex_need_nei),edited_vkey
    def accumulate_one_vertex_area(self,vkey,direction=True):
        edited_vkey=set()
        vkeys_need=[vkey]
        touch_edge_all=False
        local_max_=[]
        start=True
        while len(vkeys_need)>0:

            for vkey_ in vkeys_need:
                vkey_org_field=self.scalar_field[vkey_]
                if start:
                    touch_edge,if_editted=self.accumulate_one_vertex(vkey_,direction,False)
                    start=False
                else:
                    print(1)
                    touch_edge,if_editted=self.accumulate_one_vertex(vkey_,direction,False)
                    print(2)
                vkeys_need.remove(vkey_)
                if touch_edge==True:
                    touch_edge_all=True
                    edited_vkey.add(vkey_)
                    #return touch_edge_all,edited_vkey
                elif if_editted:
                    edited_vkey.add(vkey_)
                    neibors=self.mesh.vertex_neighbors(vkey_)
                    neibors.sort(key=lambda x: self.scalar_field[x])
                    if not direction:
                        neibors.reverse()
                    # pass_=False
                    # for neibor in neibors:   
                    #     if True or (pass_) or (self.scalar_field[neibor] > vkey_org_field and direction) or  (self.scalar_field[neibor] <vkey_org_field and not direction): 
                    #         pass_=True
                    #         if (True or neibor not in edited_vkey) and (neibor not in vkeys_need) and (neibor not in self.known_values):  
                                
                    #             vkeys_need.append(neibor)
                    changed_vkeys=self.ve_relationship.get_influenced_vertices(vkey_,direction)
                    print(vkey,vkey_,changed_vkeys)
                    vkeys_need.extend(changed_vkeys)
                    # if not pass_:
                    #     local_max_.append(neibor)
                        
                    #     if neibor not in edited_vkey: 
                    #         vkeys_need.append(neibors[-1])
        if local_max_:
            still_max=[]
            for local_max in local_max_:
                neibors=self.mesh.vertex_neighbors(local_max)
                neibors.sort(key=lambda x: self.scalar_field[x])
                if not direction:
                    neibors.reverse()
                pass_=False
                for neibor in neibors:   
                    if pass_ or (self.scalar_field[neibor] > vkey_org_field and direction) or  (self.scalar_field[neibor] <vkey_org_field and not direction): 
                        pass_=True
                if not pass_:
                    still_max.append(local_max)
            if still_max:
                for vertex in self.mesh.vertices():
                    self.mesh.vertex[vertex]['test']=self.scalar_field[vertex]
                for local_max in still_max:
                    kill_local_critical(self.mesh,local_max,'test',slope_need=0.00001,direction=True)
                print(vkey,'local_max_',still_max)
        if touch_edge_all:
            # print('touch boundary:',vkey,edited_vkey)
            self.ve_relationship.restore_back_up()
            for vkey_ in edited_vkey:
                self.scalar_field[vkey_]=self.scalar_field_org[vkey_]

        return touch_edge_all,edited_vkey                    
        
    def accumulate_one_vertex(self,vkey,direction=True,edge_searching=True):
        """
        imput 
        vkey: vertex key
        edited_vkey: vertex keys that have been edited will add vkey to this list if edited
        direction: True: accumulate from min to max, False: accumulate from max to min

        return: if this vertex touch boundary, return True, else return False
                if this vertex is edited return True, else return False
        
        
        """
        edited=False
        if direction:
            if vkey in self.up_cluster_neighbors:
                return True,False
            if edge_searching and vkey in self.ve_relationship.vertex_influenced_from_low:
                self.find_veky_gradient(vkey)

        else:
            if vkey in self.down_cluster_neighbors:
                return True,False
            if edge_searching and vkey in self.ve_relationship.vertex_influenced_from_up:
                self.find_veky_gradient(vkey)
        # #print(vkey,tuple_list)
        # neibor_edge_gras=[x[1] for x in tuple_list]
        # neibor_edge_dsfs=[x[2] for x in tuple_list]
        # neibor_edge_xyzs=[x[3] for x in tuple_list]
        # # min means the point above center with the rapidist slope
        # min_index = np.argmin(neibor_edge_gras)
        # neibor_edge_min = neibor_edge_list[min_index]
        # # max means the point below center with the rapidist slope
        # max_index = np.argmax(neibor_edge_gras)
        # neibor_edge_max = neibor_edge_list[max_index]
        
        if direction:          
            
            neibor_dsf,neibor_xyz,_=self.gradients_from_low[vkey]
            ddsf=get_ddsf_xyz(neibor_dsf,neibor_xyz,self.gra_min,self.min_dist,direction)
            #print('ddsfu:',ddsf,vkey,neibor_dsf,neibor_xyz,self.gra_min,neibor_xyz*self.gra_min-neibor_dsf)
            if ddsf is not None:
                self.scalar_field[vkey]+=ddsf 
                if self.scalar_field[vkey]>1:
                    return True,True
                edited=True
                self.ve_relationship.update_influence(vkey,direction)
              
           

        else:
            neibor_dsf,neibor_xyz,_=self.gradients_from_up[vkey]
            ddsf=get_ddsf_xyz(neibor_dsf,neibor_xyz,self.gra_min,self.min_dist,direction)
            #print('ddsfd:',ddsf,vkey,neibor_dsf,neibor_xyz,self.gra_min,neibor_xyz*self.gra_min-neibor_dsf)
            if ddsf is not None:       
                self.scalar_field[vkey]+=ddsf
                if self.scalar_field[vkey]<0:
                    return True,True
                edited=True 

                self.ve_relationship.update_influence(vkey,direction)

 
        return False,edited
  
    
    def find_veky_gradient(self,vkey):
        """
        vkey: vertex_key
        
        return: list of tuple(Ds,gradient,ds,xyz) for every neibor edge
        Ds: the fastest changing direction's point on BC D's field_value 
        ds: the distance from the center point to the D (As-Ds)
        positive: A is above D
        negative: A is below D
        xyz: the distance from the center point to D
        gradient: ds/xyz 
        
        """
        neibor_list=self.mesh.vertex_neighbors(vkey,True)
        neibor_edge_list=[{neibor_list[x-1],neibor_list[x]} for x in range(len(neibor_list))]
        tuple_list=[find_intersection_gra_4D([
                                                self.mesh.vertex_coordinates(vkey),
                                                self.mesh.vertex_coordinates(e1),
                                                self.mesh.vertex_coordinates(e2)
                                            ],
                                            [
                                                self.scalar_field[vkey],
                                                self.scalar_field[e1],
                                                self.scalar_field[e2]
                                             ]) for e1,e2 in neibor_edge_list]
        neibor_edge_gras=[x[1] for x in tuple_list]
        neibor_edge_dsfs=[x[2] for x in tuple_list]
        neibor_edge_xyzs=[x[3] for x in tuple_list]
        # min means the point above center with the rapidist slope
        min_index = np.argmin(neibor_edge_gras)
        neibor_edge_min = neibor_edge_list[min_index]
        # max means the point below center with the rapidist slope
        max_index = np.argmax(neibor_edge_gras)
        neibor_edge_max = neibor_edge_list[max_index]
        self.ve_relationship.change_vertex_edge(vkey,neibor_edge_min,neibor_edge_max)
        self.gradients_from_up[vkey]=[neibor_edge_dsfs[min_index],neibor_edge_xyzs[min_index],neibor_edge_gras[min_index]]
        self.gradients_from_low[vkey]=[neibor_edge_dsfs[max_index],neibor_edge_xyzs[max_index],neibor_edge_gras[max_index]]
        

        return neibor_edge_gras[min_index],neibor_edge_gras[max_index]


        
        

def get_ddsf_xyz(neibor_dsf,neibor_xyz,min_gra,mindist=0.0000001,direction=True):
    
    if direction:   
        ddsf=min_gra*neibor_xyz-neibor_dsf    
        if ddsf>0:
            return max(mindist,(ddsf))
        else:
            return None
    else:
        ddsf=-min_gra*neibor_xyz-neibor_dsf
        if ddsf<0:
            return min(-mindist,(ddsf))
        else:
            return None

def find_intersection_gra_4D(points,field_values):
    """
    points: list of 3 points
    field_values: list of 3 field values for the 3 ponits
    return: tuple(Ds,gradient,ds,xyz)
    Ds: the fastest changing direction's point on BC D's field_value 
    ds: the distance from the center point to the D (As-Ds)
    positive: A is above D
    negative: A is below D
    xyz: the distance from the center point to D
    gradient: ds/xyz 
    
    """
    gradient=calculate_gradient_on_triangle(points, field_values)

    # 将点转换为numpy数组
    A = np.array(points[0])
    B = np.array(points[1])
    C = np.array(points[2])
    As = field_values[0]
    Bs = field_values[1]
    Cs = field_values[2]
    

    
    # 计算法向量n
   
    n_x, n_y, n_z = gradient

    
    # 计算t

    numerator = (n_x*(B[1]-A[1])+n_y*(A[0]-B[0]))
    denominator = -(n_x*(C[1]-B[1])+n_y*(B[0]-C[0]))
    
    if denominator == 0:
        #print("分母为0，无法计算t")
        #print(points,field_values)
        numerator = (n_z*(B[1]-A[1])+n_y*(A[2]-B[2]))
        denominator = -(n_z*(C[1]-B[1])+n_y*(B[2]-C[2]))    
        if denominator == 0:
            #print("分母为0，无法计算t")    
            numerator = (n_z*(B[0]-A[0])+n_x*(A[2]-B[2]))
            denominator = -(n_z*(C[0]-B[0])+n_x*(B[2]-C[2]))     
            if denominator == 0:
                print(points,field_values)
                print("个方向均无法计算") 

                return None  # 没有交点
    
    t = numerator / denominator
    #print(t,"t value")
    # 计算交点坐标
    if t <0:
        D=B
        Ds=Bs
    elif t>1:
        D=C
        Ds=Cs
    else:
        D = B + t * np.array(C-B)
        Ds = Bs+t * np.array(Cs-Bs)
    
    # 计算垂线AD与平面的夹角的正切值
    ds=As - Ds
    xyz= math.sqrt((A[0] - D[0])**2 +(A[1] - D[1])**2+(A[2] - D[2])**2)
    max_gra = ds / xyz
    
    return Ds, max_gra,ds,xyz

def get_neighbors(mesh:Mesh, vertex_list,known_field=None,round=3,unique=False)->set:
    unique_neighbors = set()

    # Step through each vertex in the list
    for vertex in vertex_list:
        #print(vertex)
        
        neighbors = mesh.vertex_neighborhood(vertex,round)
        for neighbor in neighbors:
            if  known_field is not None and (neighbor not in known_field):
                unique_neighbors.add(neighbor)
    
    # Remove vertices that are already in the input list
    if unique:
        unique_neighbors.difference_update(vertex_list)
    
    return unique_neighbors

class heat_accumulater_for_one(heat_accumulater):
    def __init__(self, mesh, scalar_field,  min_dist=1e-8):
        super().__init__(mesh, scalar_field, None, min_dist)
    def write_scalar_field_to_mesh(self,name='scalar_field'):
        for vi in self.mesh.vertices():
            self.mesh.vertex[vi][name]=self.scalar_field[vi]
    
    def get_scalar_field_from_mesh(self,name='scalar_field'):
        self.scalar_field = {vi:self.mesh.vertex_attribute(key=vi,name=name) for vi in self.mesh.vertices()}
    def accumulate_one_vertex_area(self,vkey,direction=True):
        edited_vkey=set()
        vkeys_need=deque([vkey])
        v_set = set(vkeys_need)
        touch_edge_all=False
        local_max_=[]
        start=True
        while vkeys_need:

            vkey_=vkeys_need.popleft()
            v_set.remove(vkey_)
            vkey_org_field=self.scalar_field[vkey_]
            if start:
                touch_edge,if_editted=self.accumulate_one_vertex(vkey_,direction,False)
                start=False
            else:
                #print(1)
                touch_edge,if_editted=self.accumulate_one_vertex(vkey_,direction,False)
                #print(2)
        
            #print('accumlated on: ',vkey_,if_editted)

            if if_editted:
                edited_vkey.add(vkey_)
                neibors=self.mesh.vertex_neighbors(vkey_)
                neibors.sort(key=lambda x: self.scalar_field[x])
                if not direction:
                    neibors.reverse()
                
                for neibor in neibors:
                    if neibor not in v_set:
                        vkeys_need.append(neibor)
                        v_set.add(neibor)
                        
              
                # if not pass_:
                #     local_max_.append(neibor)
                    
                #     if neibor not in edited_vkey: 
                #         vkeys_need.append(neibors[-1])
        if local_max_:
            still_max=[]
            for local_max in local_max_:
                neibors=self.mesh.vertex_neighbors(local_max)
                neibors.sort(key=lambda x: self.scalar_field[x])
                if not direction:
                    neibors.reverse()
                pass_=False
                for neibor in neibors:   
                    if pass_ or (self.scalar_field[neibor] > vkey_org_field and direction) or  (self.scalar_field[neibor] <vkey_org_field and not direction): 
                        pass_=True
                if not pass_:
                    still_max.append(local_max)
            if still_max:
                for vertex in self.mesh.vertices():
                    self.mesh.vertex[vertex]['test']=self.scalar_field[vertex]
                for local_max in still_max:
                    kill_local_critical(self.mesh,local_max,'test',slope_need=0.00001,direction=True)
                print(vkey,'local_max_',still_max)
        # if touch_edge_all:
        #     # print('touch boundary:',vkey,edited_vkey)
        #     # self.ve_relationship.restore_back_up()
        #     for vkey_ in edited_vkey:
        #         self.scalar_field[vkey_]=self.scalar_field_org[vkey_]

        return touch_edge_all,edited_vkey     
    def accumulate_one_vertex(self,vkey,direction=True,edge_searching=True):
        """
        imput 
        vkey: vertex key
        edited_vkey: vertex keys that have been edited will add vkey to this list if edited
        direction: True: accumulate from min to max, False: accumulate from max to min

        return: if this vertex touch boundary, return True, else return False
                if this vertex is edited return True, else return False
        
        
        """
        edited=False
        if direction:
            if vkey in self.up_cluster_neighbors:
                return True,False
            if edge_searching:
                self.find_veky_gradient(vkey)

        else:
            if vkey in self.down_cluster_neighbors:
                return True,False
            if edge_searching :
                self.find_veky_gradient(vkey)
        # #print(vkey,tuple_list)
        # neibor_edge_gras=[x[1] for x in tuple_list]
        # neibor_edge_dsfs=[x[2] for x in tuple_list]
        # neibor_edge_xyzs=[x[3] for x in tuple_list]
        # # min means the point above center with the rapidist slope
        # min_index = np.argmin(neibor_edge_gras)
        # neibor_edge_min = neibor_edge_list[min_index]
        # # max means the point below center with the rapidist slope
        # max_index = np.argmax(neibor_edge_gras)
        # neibor_edge_max = neibor_edge_list[max_index]
        
        if direction: 
            #print(vkey)         
            
            neibor_dsf,neibor_xyz,_=self.find_veky_gradient(vkey)
            ddsf=get_ddsf_xyz(neibor_dsf,neibor_xyz,self.gra_min,self.min_dist,direction)
            #print('ddsfu:',ddsf,vkey,neibor_dsf,neibor_xyz,self.gra_min,neibor_xyz*self.gra_min-neibor_dsf)
            if ddsf is not None:
                self.scalar_field[vkey]+=ddsf 
                # if self.scalar_field[vkey]>1:
                #     return True,True
                edited=True
                
              
           

        else:
            neibor_dsf,neibor_xyz,_=self.gradients_from_up[vkey]
            ddsf=get_ddsf_xyz(neibor_dsf,neibor_xyz,self.gra_min,self.min_dist,direction)
            #print('ddsfd:',ddsf,vkey,neibor_dsf,neibor_xyz,self.gra_min,neibor_xyz*self.gra_min-neibor_dsf)
            if ddsf is not None:       
                self.scalar_field[vkey]+=ddsf
                # if self.scalar_field[vkey]<0:
                #     return True,True
                edited=True 

               
        return False,edited
    def find_veky_gradient(self,vkey):
        """
        vkey: vertex_key
        
        return: list of tuple(Ds,gradient,ds,xyz) for every neibor edge
        Ds: the fastest changing direction's point on BC D's field_value 
        ds: the distance from the center point to the D (As-Ds)
        positive: A is above D
        negative: A is below D
        xyz: the distance from the center point to D
        gradient: ds/xyz 
        
        """
        #neibor_list=self.mesh.vertex_neighbors(vkey,True)
        neibor_face_list=list(self.mesh.vertex_faces(vkey))
        neibor_face_vertices_list=[list(self.mesh.face_vertices(face)) for face in neibor_face_list]
        #print(neibor_face_vertices_list)
        neibor_edge_list=[sublist[:sublist.index(vkey)] + sublist[sublist.index(vkey) + 1:] for sublist in neibor_face_vertices_list]
        #print(neibor_edge_list)
   
        tuple_list=[find_intersection_gra_4D([
                                                self.mesh.vertex_coordinates(vkey),
                                                self.mesh.vertex_coordinates(e1),
                                                self.mesh.vertex_coordinates(e2)
                                            ],
                                            [
                                                self.scalar_field[vkey],
                                                self.scalar_field[e1],
                                                self.scalar_field[e2]
                                             ]) for e1,e2 in neibor_edge_list]
        neibor_edge_gras=[x[1] for x in tuple_list]
        neibor_edge_dsfs=[x[2] for x in tuple_list]
        neibor_edge_xyzs=[x[3] for x in tuple_list]
        # min means the point above center with the rapidist slope
        min_index = np.argmin(neibor_edge_gras)
        neibor_edge_min = neibor_edge_list[min_index]
        # max means the point below center with the rapidist slope
        max_index = np.argmax(neibor_edge_gras)
        neibor_edge_max = neibor_edge_list[max_index]
        #self.ve_relationship.change_vertex_edge(vkey,neibor_edge_min,neibor_edge_max)
        self.gradients_from_up[vkey]=[neibor_edge_dsfs[min_index],neibor_edge_xyzs[min_index],neibor_edge_gras[min_index]]
        self.gradients_from_low[vkey]=[neibor_edge_dsfs[max_index],neibor_edge_xyzs[max_index],neibor_edge_gras[max_index]]
        

        return self.gradients_from_low[vkey]

 
        
class vertex_edge_relationship:
    def __init__(self,mesh:Mesh,known_field:Dict):
        self.mesh=mesh
        self.vertex_upper_edge={}
        self.vertex_lower_edge={}
        self.edge_upper_vertex={}
        self.edge_lower_vertex={}
        self.vertex_influenced_from_low=set()
        self.vertex_influenced_from_up=set()
        for vertex in mesh.vertices():
            self.vertex_upper_edge[vertex]=tuple()
            self.vertex_lower_edge[vertex]=tuple()
        for edge in mesh.edges():
            if edge[0]>edge[1]:
                edge=(edge[1],edge[0])
            self.edge_lower_vertex[edge]=-1
            self.edge_upper_vertex[edge]=-1
    
    def save_back_up(self):
        self.edge_lower_vertex_back_up= self.edge_lower_vertex.copy()
        self.edge_upper_vertex_back_up= self.edge_upper_vertex.copy()
        self.vertex_upper_edge_back_up=self.vertex_upper_edge.copy()
        self.vertex_lower_edge_back_up=self.vertex_lower_edge.copy()
        self.vertex_influenced_from_low_back_up=self.vertex_influenced_from_low.copy()
        self.vertex_influenced_from_up_back_up=self.vertex_influenced_from_up.copy()
    
    def restore_back_up(self):
        self.edge_lower_vertex=self.edge_lower_vertex_back_up
        self.edge_upper_vertex=self.edge_upper_vertex_back_up
        self.vertex_upper_edge=self.vertex_upper_edge_back_up
        self.vertex_lower_edge=self.vertex_lower_edge_back_up
        self.vertex_influenced_from_low=self.vertex_influenced_from_low_back_up
        self.vertex_influenced_from_up=self.vertex_influenced_from_up_back_up
        

    def change_vertex_edge(self,vkey:int,edge_upper:Tuple,edge_lower:Tuple):
        """
        edge upper: edge higher than vkey with largest gradient normal
        edge lower: edge lower than vkey with largest gradient normal
        """
        #print(edge_upper,edge_lower)
        edge_lower=get_edge_tuple(edge_lower)
        edge_upper=get_edge_tuple(edge_upper)
        
        upper_edge_o= self.vertex_upper_edge[vkey]
        if upper_edge_o and upper_edge_o!=edge_upper:
            self.edge_upper_vertex[upper_edge_o]=-1
        lower_edge_o= self.vertex_lower_edge[vkey]
        if lower_edge_o and lower_edge_o!=edge_lower:
            self.edge_lower_vertex[lower_edge_o]=-1

        self.vertex_upper_edge[vkey]=edge_upper
        self.vertex_lower_edge[vkey]=edge_lower
        
        self.edge_upper_vertex[edge_upper]=vkey
        self.edge_lower_vertex[edge_lower]=vkey
    

    def update_influence(self,vkey,direction):
        if direction:
            self.vertex_influenced_from_low.discard(vkey)
        else:    
            self.vertex_influenced_from_up.discard(vkey)
    
        for edge in self.mesh.vertex_edges(vkey):

            self.vertex_influenced_from_low.add(self.edge_upper_vertex[edge])
            self.vertex_influenced_from_up.add( self.edge_lower_vertex[edge])
    def get_influenced_vertices(self,vkey,direction)->List:
        if_v=[]
        if direction:
            for edge in self.mesh.vertex_edges(vkey):
                if self.edge_upper_vertex[edge]>=0:

                    if_v.append(self.edge_upper_vertex[edge])
        else:
            for edge in self.mesh.vertex_edges(vkey):
                if self.edge_lower_vertex[edge]>=0:

                    if_v.append(self.edge_lower_vertex[edge])
        return if_v

    
    def get_upper_edge(self,vkey:int)->Set:
        return set(self.vertex_upper_edge[vkey])
    
    def get_lower_edge(self,vkey:int)->Set:
        return set(self.vertex_lower_edge[vkey])
    
    def get_upper_vertex(self,edge:Set)->Set:
        edge=(list(edge)).sort()
        return set(self.edge_upper_vertex[edge])
    
    def get_lower_vertex(self,edge:Set)->Set:
        edge=(list(edge)).sort()
        return set(self.edge_lower_vertex[edge])
    
def get_edge_tuple(edge:Tuple):
    a,b=edge
    if a>b:
        return (b,a)
    else:
        return (a,b)    
def get_face_gradient_from_scalar_field(mesh:Mesh, u, use_igl=True):
    """
    Finds face gradient from scalar field u.
    Scalar field u is given per vertex.

    Parameters
    ----------
    mesh: :class: 'compas.datastructures.Mesh'
    u: list, float. (dimensions : #VN x 1)

    Returns
    ----------
    np.array (dimensions : #F x 3) one gradient vector per face.
    """
    print('Computing per face gradient')
    if use_igl:
        try:
            import igl
            v, f = mesh.to_vertices_and_faces()
            G = igl.grad(np.array(v), np.array(f))
            X = G * u
            nf = len(list(mesh.faces()))
            X = np.array([[X[i], X[i + nf], X[i + 2 * nf]] for i in range(nf)])
            return X
        except ModuleNotFoundError:
            print("Could not calculate gradient with IGL because it is not installed. Falling back to default function")

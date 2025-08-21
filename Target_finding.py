from compas.datastructures import Mesh
from compas_slicer.pre_processing import create_mesh_boundary_attributes
import numpy as np
class Target_finder:
    def __init__(self,mesh:Mesh):
        self.mesh = mesh
    def output(self):
        """
        output:
        clustered_high_boundary_vs
        clustered_low_bounary_vs
        """
        self.find()
        self.separate_high_and_low()
        create_mesh_boundary_attributes(self.mesh,self.low_boundary_vs,self.high_boundary_vs)
        return self.high_boundary_vs,self.low_boundary_vs
    
    def find(self):
        naked=set()
        naked_edges = set()
        for edge in self.mesh.edges():
            faces=self.mesh.edge_faces(edge[0],edge[1])
            if None in faces:
                naked.update(edge)
                naked_edges.add(edge)
            
        boundary_vs = list(naked)
     
        #print(boundary_vs)
        self.cluster_vs = group_connected_elements(boundary_vs, naked_edges)
        print(len(self.cluster_vs))
        
    
    def separate_high_and_low(self):
        self.high_boundary_vs = []
        self.low_boundary_vs = []
        for boundary in self.cluster_vs:
            avg_vector = avg_normalized_neighbor_vector(self.mesh,boundary)
            if avg_vector[2]>0:
                self.low_boundary_vs.append(boundary)
            else:
                self.high_boundary_vs.append(boundary)
        



def group_connected_elements(nums, pairs):
    from collections import defaultdict
    # 初始化并查集
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # 路径压缩
            x = parent[x]
        return x
    
    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x
    
    # 处理所有对，建立连通关系
    for a, b in pairs:
        union(a, b)
    
    # 将数字分组
    groups = defaultdict(list)
    for num in nums:
        root = find(num)
        groups[root].append(num)
    
    return list(groups.values())

def avg_normalized_neighbor_vector(
    mesh:Mesh,
    vkeys) :
    """
    计算给定 vkeys 的邻居单位化向量的平均值。

    参数
    ----
    mesh : 提供以下接口的对象
        - mesh.neibour(vkey) -> Iterable[int]  # 注意你给的拼写 'neibour'
        - mesh.coordinate(vkey) -> [x, y, z]
    vkeys : 需要处理的顶点索引集合
    返回所有顶点合在一起的全局平均向量 [x, y, z]。

    返回
    ----
    List[float] 
    """
    def coord(vk: int) -> np.ndarray:
        return np.asarray(mesh.vertex_coordinates(vk), dtype=float)

    eps = 1e-12  # 防止除零
    vkeys = list(vkeys)


    all_units = []
    for vk in vkeys:
        c = coord(vk)
        for nb in mesh.vertex_neighbors(vk):
            vec = coord(nb) - c
            nrm = np.linalg.norm(vec)
            if nrm > eps:
                all_units.append(vec / nrm)

    if not all_units:
        return [0.0, 0.0, 0.0]

    mean_vec = np.mean(np.vstack(all_units), axis=0)
    return mean_vec.tolist()

def main():
    import os


    input_folder_name='beam1A'
    DATA_PATH = os.path.join(os.path.dirname(__file__), input_folder_name)
    OBJ_INPUT_NAME = os.path.join(DATA_PATH, 'mesh.obj')
    
    # --- Load initial_mesh

    mesh = Mesh.from_obj(os.path.join(DATA_PATH, OBJ_INPUT_NAME))
    #print(mesh)
    t_f = Target_finder(mesh)
    print(t_f.output())

if __name__ == "__main__":
    main()
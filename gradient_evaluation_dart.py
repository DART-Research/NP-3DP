import numpy as np
import math
import logging
import compas_slicer.utilities as utils
#from compas_slicer.pre_processing.preprocessing_utils import get_face_edge_vectors
from compas_slicer.pre_processing.preprocessing_utils import get_vertex_gradient_from_face_gradient
from compas.datastructures import Mesh
#This apart is add by Yichuan
import gudhi
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from compas_slicer.pre_processing.gradient_evaluation import GradientEvaluation
from mesh_changing import get_unique_neighbors
from gradient_descent import kill_local_criticals,kill_local_critical
from gradient_optimization import heat_accumulater_for_one
#End

logger = logging.getLogger('logger')

__all__ = ['GradientEvaluation_Dart']

class GradientEvaluation_Dart(GradientEvaluation):
    def __init__(self, mesh: Mesh, DATA_PATH,use_igl=True):
        logger.info('Gradient evaluation')
        self.use_igl=use_igl
        self.mesh = mesh
        self.DATA_PATH = DATA_PATH
        self.OUTPUT_PATH = utils.get_output_directory(DATA_PATH)

        self.minima, self.maxima, self.saddles = [], [], []
        ### This is add by YIchuan
        self.repeatpoint = []
        ### end
        self.face_gradient = []  # np.array (#F x 3) one gradient vector per face.
        self.vertex_gradient = []  # np.array (#V x 3) one gradient vector per vertex.
        self.face_gradient_norm = []  # list (#F x 1)
        self.vertex_gradient_norm = []  # list (#V x 1)

     ### This vectors are writen by YIchuan

    def get_neighbors(self, point):
        # 获取直接相邻点
        direct_neighbors = self.mesh.vertex_neighbors(point)
        all_neighbors = set(direct_neighbors)

        # 获取相邻点的相邻点
        for neighbor in direct_neighbors:
            second_layer_neighbors = self.mesh.vertex_neighbors(neighbor)
            all_neighbors.update(second_layer_neighbors)

        # 排除自身
        all_neighbors.discard(point)
        return list(all_neighbors)
    
    def are_fields_equal(self, field1, field2, tolerance):
        # 假设field值是数值类型，如果是其他类型，需要修改比较逻辑
        return abs(field1 - field2) <= tolerance
    # def compute_gradient(self):
    #     """ Computes the gradient on the faces and the vertices. """
    #     u_v = [self.mesh.vertex[vkey]['scalar_field'] for vkey in self.mesh.vertices()]
    #     if isinstance (u_v[0],list):
    #         print("cannot get gradient")

    #     else:
            
    #         self.face_gradient = get_face_gradient_from_scalar_field(self.mesh, u_v)
    #         self.vertex_gradient = get_vertex_gradient_from_face_gradient(self.mesh, self.face_gradient)
    def compute_gradient(self):
        """ Computes the gradient on the faces and the vertices. """
     
        u_v = []
        for vkey in self.mesh.vertices():
 
            
            u_v.append(self.mesh.vertex[vkey]['scalar_field'])


                
        if isinstance (u_v[0],list):
            print("cannot get gradient")

        else:

                        
            self.face_gradient = get_face_gradient_from_scalar_field(self.mesh, u_v,use_igl=self.use_igl)
            self.edge_gradient = get_edge_gradient_from_face_gradient(self.mesh, self.face_gradient)
            self.vertex_gradient = get_vertex_gradient_from_face_gradient(self.mesh, self.face_gradient)
    def return_scalar_field_dictionary(self):
        """ Returns a dictionary with the scalar field values for each vertex. """
        logger.info('Returning scalar field dictionary')
        return {vkey: self.mesh.vertex[vkey]['scalar_field'] for vkey in self.mesh.vertices()}
    def compute_gradient_norm(self):
        """ Computes the norm of the gradient. """
        first_vertex=list(self.mesh.vertices())[0]
        if isinstance( self.mesh.vertex[first_vertex]['scalar_field'],list):
            print("cannot get norm")
        else:
            logger.info('Computing norm of gradient')
            f_g = np.array([self.face_gradient[i] for i, fkey in enumerate(self.mesh.faces())])
            v_g = np.array([self.vertex_gradient[i] for i, vkey in enumerate(self.mesh.vertices())])
            self.face_gradient_norm = list(np.linalg.norm(f_g, axis=1))
            self.vertex_gradient_norm = list(np.linalg.norm(v_g, axis=1))
    def compute_gradient_norm_max_neighbor_faces(self):
        """ Computes the norm of the gradient. 
         
        Finds vertex gradient given an already calculated per face gradient.

        Parameters
        ----------
        mesh: :class: 'compas.datastructures.Mesh'
        face_gradient: np.array with one vec3 per face of the mesh. (dimensions : #F x 3)

        Returns
        ----------
        np.array (dimensions : #V x 3) one gradient vector per vertex.

        """
        faces=list(self.mesh.faces())
        vertices=list(self.mesh.vertices())
        logger.info('Computing norm of gradient based on face g max')
        mean=np.mean(self.face_gradient_norm)
        mesh=self.mesh
        face_gradient=self.face_gradient
        face_gradient_dict={}
        for i,face_id in enumerate(mesh.faces()):
            face_gradient_dict[face_id]=face_gradient[i]
        logger.info('Computing per vertex gradient')
        vertex_gradient = []
        for v_key in mesh.vertices():
            choose_list=[]
            choose_list.append(self.vertex_gradient_norm[vertices.index(v_key)])
            face_gradient_max=0
            total_area=0
            total_gradient=0
            all_face_gre_v=[]
            for f_key in mesh.vertex_faces(v_key):
                face_gradient_norm=self.face_gradient_norm[faces.index(f_key)]
                
                if not math.isnan(face_gradient_norm):
                    face_area=mesh.face_area(f_key)
                    total_area+=face_area
                    total_gradient+=face_gradient_norm*face_area
                    all_face_gre_v.append(face_gradient_norm)
                else:
                    print("Error",v_key,f_key,face_gradient_dict[f_key])
                
            try:
                v_grad = total_gradient/total_area
                if not math.isnan(v_grad):
                    
                    choose_list.append(v_grad)
                else:
                    print("Error1",v_key)   

            except:
                print("Error",v_key,[mesh.vertex_faces(v_key)],[mesh.face_area(f_key) for f_key in mesh.vertex_faces(v_key)])
                # v_grad=np.mean([np.linalg.norm(face_gradient_dict[f_key]) for f_key in mesh.vertex_faces(v_key)])
                # print("gradient",v_grad)
                # if not math.isnan(v_grad):
                #     choose_list.append(v_grad)
                # else:
                #     print("Error2",v_key) 
            
            try:
                mid_gradient=np.median(all_face_gre_v)
                if math.isnan(mid_gradient):
                    
                    print("Error4",v_key,all_face_gre_v)
                choose_list.append(mid_gradient)
            except:
                print("Error3",v_key,all_face_gre_v)
           
                
            vertex_gradient.append(find_closest_value(choose_list,mean))


        

    

        self.vertex_gradient_norm =vertex_gradient
    def get_critical_points_with_same_scalar_field(self, points, tolerance=0.01):
        print("get_critical_points_with_same_scalar_field")
        unique_points = {}

        for point in points:
            field_value = self.mesh.vertex_attributes(point)['scalar_field']
            found = False
            for existing_field in unique_points:
                if self.are_fields_equal(field_value, existing_field, tolerance):
                    found = True
                    unique_points[existing_field].append(point)
                    break
            if not found:
                unique_points[field_value] = [point]

        connected_points = []

        for field_value, point_list in unique_points.items():
            connected_components = self.find_connected_components(point_list)
            # 打印connected_components的数量和内容
            print(f"Field value: {field_value}")
            print(f"Number of connected components: {len(connected_components)}")
            for i, component in enumerate(connected_components):
                print(f"Component {i}: {component}")

            connected_points.extend(self.select_one_point_from_each_connected_component(connected_components))

        return connected_points

    def find_connected_components(self, point_list):
        visited = set()
        connected_components = []
        point_set = set(point_list)  # 转换为集合，便于快速查找

        for point in point_list:
            if point not in visited:
                connected_component = self.dfs(point, visited, point_set)
                connected_components.append(connected_component)

        return connected_components

    def dfs(self, start_point, visited, point_set):
        stack = [start_point]
        connected_component = []

        while stack:
            current_point = stack.pop()
            if current_point not in visited:
                visited.add(current_point)
                connected_component.append(current_point)
                neighbors = self.mesh.vertex_neighborhood(current_point,3)  # 使用你的 get_neighbors 函数获取相邻点
                for neighbor in neighbors:
                    if neighbor in point_set and neighbor not in visited:
                        stack.append(neighbor)

        return connected_component


    def select_one_point_from_each_connected_component(self, connected_components):
        selected_points = []

        for component in connected_components:
            if component:  # 如果连通分量不为空
                selected_points.append(component[0])  # 只选择连通分量中的第一个点

        return selected_points
    
    def compute_critical_points_connection(self):
        """
        This is a failure of writteing reeb graph
        """
        field_values = [data['scalar_field'] for v_key, data in self.mesh.vertices(data=True)]
        local_maxima = self.maxima
        local_minima = self.minima
        saddles = self.saddles

        # 创建一个空的连接矩阵
        num_points = len(field_values)
        connectivity_matrix = np.zeros((num_points, num_points))

        # 打印配对信息
        print("Saddle points and their neighboring local maxima and minima:")
        for saddle in saddles:
            neighboring_maxima = [local_max for local_max in local_maxima if local_max > saddle]
            neighboring_minima = [local_min for local_min in local_minima if local_min < saddle]

            pairs = [(saddle, max_val) for max_val in neighboring_maxima] + [(saddle, min_val) for min_val in neighboring_minima]
            print(f"Saddle point {saddle} is connected to:")
            for pair in pairs:
                print(pair)

        # 遍历每一个鞍点
        for saddle in saddles:
            # 找到与鞍点相邻的局部最大值和局部最小值的索引
            neighboring_maxima = [local_max for local_max in local_maxima if local_max > saddle]
            neighboring_minima = [local_min for local_min in local_minima if local_min < saddle]

            # 连接鞍点与相邻的局部最大值和局部最小值
            for max_value in neighboring_maxima:
                max_index = max_value
                connectivity_matrix[saddle, max_index] = 1
                connectivity_matrix[max_index, saddle] = 1
            for min_value in neighboring_minima:
                min_index = min_value
                connectivity_matrix[saddle, min_index] = 1
                connectivity_matrix[min_index, saddle] = 1

        return connectivity_matrix 



    def find_critical_points(self):
        """ Finds minima, maxima and saddle points of the scalar function on the mesh. """
        start=True
        self.saddles = []
        self.maxima = []    
        self.minima = []
        for vkey, data in self.mesh.vertices(data=True):
       
            boundary = self.mesh.vertex_attribute(key=vkey,name='boundary')
            if boundary==1 or boundary ==2:
        
                continue

            
            current_v = data['scalar_field']
            neighbors = self.mesh.vertex_neighbors(vkey, ordered=True)
            if isinstance(current_v,list):
                if start:
                    self.saddles = [[] for _ in range(len(current_v))]
                    start=False
                for i,current_vi in enumerate(current_v):
                    #print(i)
                    values = []
                    if len(neighbors) > 0:
                        neighbors.append(neighbors[0])
                      
                        for n in neighbors:
                            v = self.mesh.vertex_attributes(n)['scalar_field'][i]
                            if abs(v - current_vi) > 0.0:
                                values.append(current_vi - v)
                            else:
                                print('error',vkey)
                        sgc = count_sign_changes(values)

                        #print(vkey,sgc)
                        if sgc > 2:
                            if sgc % 2 == 0:
                                print(vkey,'find saddle point')
                                self.saddles[i].append(vkey)    
                            else:
                                print(vkey,'find saddle point ?',sgc)
                                self.saddles[i].append(vkey)

            else:
                neighbors = self.mesh.vertex_neighbors(vkey, ordered=True)
                values = []
                if len(neighbors) > 0:
                    neighbors.append(neighbors[0])
                    for n in neighbors:
                        v = self.mesh.vertex_attributes(n)['scalar_field']
                        if abs(v - current_v) > 0.0:
                            values.append(current_v - v)
                        else:
                            print('error',vkey)
                    sgc = count_sign_changes(values)
                    

                    if sgc == 0:  # extreme point
                        if current_v > min(self.mesh.vertex_attributes((neighbor))['scalar_field'] for neighbor in neighbors):
                            self.maxima.append(vkey)
                            print(vkey,'find maxima point')
                        else:
                            self.minima.append(vkey)
                            print(vkey,'find minima point')
                    # if sgc == 2:  # regular point
                    #     pass
                    # if vkey==9892:
                    #     print(vkey,sgc)
                    if sgc > 2 :
                        if sgc % 2 == 0:
                            print(vkey,'find saddle point')
                            self.saddles.append(vkey)
                            #print(self.saddles)
                        else:
                            print(vkey,'find saddle point ?',sgc)
                            self.saddles[i].append(vkey)

                
        # self.maxima=self.get_critical_points_with_same_scalar_field(self.maxima)
        # self.minima=self.get_critical_points_with_same_scalar_field(self.minima)
        try:
            self.saddles=self.delet_multiple_point(self.saddles,round=3)
        except:
            for i,saddles in enumerate(self.saddles):
                self.saddles[i]=self.delet_multiple_point(saddles,round=3)

        #print(len(self.maxima),len(self.minima),len(self.saddles))
        #self.compute_critical_points_connection()
    def find_critical_points_with_related_boundary(self):
        """ Finds minima, maxima and saddle points of the scalar function on the mesh. """
        start=True
        self.related_boundary={}
        for vkey, data in self.mesh.vertices(data=True):
       
            boundary = self.mesh.vertex_attribute(key=vkey,name='boundary')
            if boundary==1 or boundary ==2:
        
                continue

            
            current_v = data['scalar_field']
            neighbors = self.mesh.vertex_neighbors(vkey, ordered=True)


    
            neighbors = self.mesh.vertex_neighbors(vkey, ordered=True)
            values = []
            if len(neighbors) > 0:
                neighbors.append(neighbors[0])
                for n in neighbors:
                    v = self.mesh.vertex_attributes(n)['scalar_field']
                    if abs(v - current_v) > 0.0:
                        values.append(current_v - v)
                    else:
                        print('error',vkey)
                sgc,max_v,min_v = count_sign_changes_with_vkey(values)
                max_v = [neighbors[i] for i in max_v]
                min_v = [neighbors[i] for i in min_v]
                if sgc > 2 :
                    self.related_boundary[vkey]={}
                    if sgc % 2 == 0:
                        print(vkey,'find saddle point',max_v,min_v)
                        self.saddles.append(vkey)
                        self.related_boundary[vkey]['max_v']=max_v
                        self.related_boundary[vkey]['min_v']=min_v
                        #print(self.saddles)
                    else:
                        print(vkey,'find saddle point ????????',sgc)
                        self.saddles.append(vkey)
                        self.related_boundary[vkey]['max_v']=max_v
                        self.related_boundary[vkey]['min_v']=min_v
       
    def kill_max_min(self):
        maxim=[i for i in self.maxima if self.mesh.vertex_attribute(key=i,name='boundary')!=2]
        minim=[i for i in self.minima if self.mesh.vertex_attribute(key=i,name='boundary')!=1]
        print('sorted max',maxim)
        print('sorted min',minim)
        kill_local_criticals(self.mesh,maxim,minim,name='scalar_field',slope_need=0.00001)
        self.find_critical_points() 
        if self.maxima or self.minima: 
            self.kill_max_min()          
    def kill_max_min_muliti_way(self):
        maxim=[i for i in self.maxima if self.mesh.vertex_attribute(key=i,name='boundary')!=2]
        minim=[i for i in self.minima if self.mesh.vertex_attribute(key=i,name='boundary')!=1]
        print('sorted max',maxim)
        print('sorted min',minim)
        heat_accumulating = heat_accumulater_for_one(self.mesh,{})
        for point in maxim:
            kill_local_critical(self.mesh,point,'scalar_field',0.0000001,True)
        heat_accumulating.get_scalar_field_from_mesh()    
        for point in minim:
            print(point,self.mesh.vertex[point]['scalar_field'])
            
            heat_accumulating.gra_min=0.0000001
            heat_accumulating.min_dist=0.001
            heat_accumulating.accumulate_one_vertex_area(point,True)
        heat_accumulating.write_scalar_field_to_mesh()
        print([self.mesh.vertex[point]['scalar_field'] for point in minim])
        self.find_critical_points()
        if self.maxima or self.minima:
            self.kill_max_min_muliti_way()
               

    def delet_multiple_point(self,veky_list,round=2):
        org_list=[x for x in veky_list]
        out_list=[]
        for i in range(len(veky_list)):
            if org_list:
                if not out_list:
                    out_list.append(org_list[0])
                    org_list.pop(0)
                else:
                    neibors=get_unique_neighbors(mesh=self.mesh,vertex_list=out_list,round=round)
                    if not (org_list[0] in neibors):
                        out_list.append(org_list[0])
                    org_list.pop(0)
        return out_list
    


# The below is written by ioanna
def count_sign_changes(values):
    """ Returns the number of sign changes in a list of values. """
    count = 0
    prev_v = 0
    for i, v in enumerate(values):
        if i == 0:
            prev_v = v
        else:
            if prev_v * v < 0:
                count += 1
            prev_v = v
    return count
# The below is written by Yichuan
def count_sign_changes_with_vkey(values):
    """ Returns the number of sign changes in a list of values. """
    count = 0
    prev_v = 0
    max_v=[]
    min_v=[]
    for i, v in enumerate(values):
        if i == 0:
            prev_v = v
        else:
            if prev_v * v < 0:

                count += 1
                if v<0:
                    max_v.append(i-1)
                else:
                    min_v.append(i-1)

            prev_v = v
    return count,max_v,min_v
# The below is written by Yichuan
def get_edge_gradient_from_face_gradient(mesh:Mesh, face_gradient):
    #print('stop')
    face_gradient_dict={}
    for i,face_id in enumerate(mesh.faces()):
        face_gradient_dict[face_id]=face_gradient[i]

    edge_gradients = {}
    for edge in mesh.edges():
        try:
            # print('edge',edge)
            # faces=mesh.edge_faces(edge[0],edge[1])
            # print('face',faces)
            # print(face_gradient_dict[faces[0]])
            edge_gradient=([face_gradient_dict[face] for face in mesh.edge_faces(edge[0],edge[1])])
            edge_gradient=np.mean(edge_gradient,axis=0)
            #print("gradient",edge_gradient)
        except:
            try:
                #print("maybe naked",edge)
                edge_gradient=(face_gradient_dict[mesh.edge_faces(edge[0],edge[1])])[0]
                #print("naked gradient", edge_gradient)
            except:
                #print("None",edge)
                faces=mesh.edge_faces(edge[0],edge[1])
                face=[x for x in faces if x is not None][0]
                edge_gradient=(face_gradient_dict[face])
                         
        edge_gradients[edge]=edge_gradient
    #print( edge_gradients)
    return edge_gradients
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
    logger.info('Computing per face gradient')
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

        grad = []
        for fkey in mesh.faces():
            A = mesh.face_area(fkey)
            N = mesh.face_normal(fkey)
            edge_0, edge_1, edge_2 = get_face_edge_vectors(mesh, fkey)
            v0, v1, v2 = mesh.face_vertices(fkey)
            u0 = u[v0]
            u1 = u[v1]
            u2 = u[v2]
            vc0 = np.array(mesh.vertex_coordinates(v0))
            vc1 = np.array(mesh.vertex_coordinates(v1))
            vc2 = np.array(mesh.vertex_coordinates(v2))
            # grad_u = -1 * ((u1-u0) * np.cross(vc0-vc2, N) + (u2-u0) * np.cross(vc1-vc0, N)) / (2 * A)
            grad_u = ((u1-u0) * np.cross(vc0-vc2, N) + (u2-u0) * np.cross(vc1-vc0, N)) / (2 * A)
            # grad_u = (np.cross(N, edge_0) * u2 +
            #           np.cross(N, edge_1) * u0 +
            #           np.cross(N, edge_2) * u1) / (2 * A)
            grad.append(grad_u)
        return np.array(grad)
def get_face_edge_vectors(mesh:Mesh, fkey):
    """ Returns the edge vectors of the face with fkey. """
    e0, e1, e2 = mesh.face_halfedges(fkey)
    edge_0 = np.array(mesh.vertex_coordinates(e0[0])) - np.array(mesh.vertex_coordinates(e0[1]))
    edge_1 = np.array(mesh.vertex_coordinates(e1[0])) - np.array(mesh.vertex_coordinates(e1[1]))
    edge_2 = np.array(mesh.vertex_coordinates(e2[0])) - np.array(mesh.vertex_coordinates(e2[1]))
    return edge_0, edge_1, edge_2
def get_vertex_gradient_from_face_gradient(mesh:Mesh, face_gradient):
    """
    Finds vertex gradient given an already calculated per face gradient.

    Parameters
    ----------
    mesh: :class: 'compas.datastructures.Mesh'
    face_gradient: np.array with one vec3 per face of the mesh. (dimensions : #F x 3)

    Returns
    ----------
    np.array (dimensions : #V x 3) one gradient vector per vertex.
    """
    face_gradient_dict={}
    for i,face_id in enumerate(mesh.faces()):
        face_gradient_dict[face_id]=face_gradient[i]
    logger.info('Computing per vertex gradient')
    vertex_gradient = []
    for v_key in mesh.vertices():
        faces_total_area = 0
        faces_total_grad = np.array([0.0, 0.0, 0.0])
        face_area_max=0
        for f_key in mesh.vertex_faces(v_key):
            face_area = mesh.face_area(f_key)
            # if face_area>face_area_max:
            #     face_area_max=face_area
            #     v_grad=face_gradient_dict[f_key]
            faces_total_area += face_area
            faces_total_grad += face_area * face_gradient_dict[f_key]
        v_grad = faces_total_grad / faces_total_area
        vertex_gradient.append(v_grad)
    return np.array(vertex_gradient)
def get_vertex_gradient_from_z(mesh:Mesh, u, use_igl=True):
    facekeys=mesh.vertex_faces(u)
    faces={key:mesh.face[key] for key in facekeys}
    vkeys=mesh.vertex_neighbors(u)
    vkeys.append(u)
    vertices={key:mesh.vertex[key] for key in vkeys}
    new_mesh=Mesh.from_vertices_and_faces(vertices,faces)
    zs=[]
    for vkey in new_mesh.vertices():
        zs.append(mesh.vertex[vkey]['z'])
    face_gra=get_face_gradient_from_scalar_field(new_mesh,zs,use_igl)
    
    def get_vertex_gradient(v_key,face_gradient):
        face_gradient_dict={}
        for i,face_id in enumerate(mesh.faces()):
            face_gradient_dict[face_id]=face_gradient[i]
        faces_total_area = 0
        faces_total_grad = np.array([0.0, 0.0, 0.0])
        for f_key in mesh.vertex_faces(v_key):
            face_area = mesh.face_area(f_key)
            faces_total_area += face_area
            faces_total_grad += face_area * face_gradient_dict[f_key]
        v_grad = faces_total_grad / faces_total_area
        return(v_grad)
    return get_vertex_gradient(u,face_gra)
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
    logger.info('Computing per face gradient')
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

        grad = []
        for fkey in mesh.faces():
            A = mesh.face_area(fkey)
            N = mesh.face_normal(fkey)
            edge_0, edge_1, edge_2 = get_face_edge_vectors(mesh, fkey)
            v0, v1, v2 = mesh.face_vertices(fkey)
            u0 = u[v0]
            u1 = u[v1]
            u2 = u[v2]
            vc0 = np.array(mesh.vertex_coordinates(v0))
            vc1 = np.array(mesh.vertex_coordinates(v1))
            vc2 = np.array(mesh.vertex_coordinates(v2))
            # grad_u = -1 * ((u1-u0) * np.cross(vc0-vc2, N) + (u2-u0) * np.cross(vc1-vc0, N)) / (2 * A)
            grad_u = ((u1-u0) * np.cross(vc0-vc2, N) + (u2-u0) * np.cross(vc1-vc0, N)) / (2 * A)
            # grad_u = (np.cross(N, edge_0) * u2 +
            #           np.cross(N, edge_1) * u0 +
            #           np.cross(N, edge_2) * u1) / (2 * A)
            grad.append(grad_u)
        return np.array(grad)
def find_closest_value(lst, a):
    # 使用 min 函数和 key 参数来找到最接近的值
    closest_value = min(lst, key=lambda x: abs(x - a))
    if math.isnan(closest_value):
        closest_value=a
    return closest_value
if __name__ == "__main__":
    pass



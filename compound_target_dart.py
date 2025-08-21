from compas_slicer.pre_processing import CompoundTarget,blend_union_list,stairs_union_list,chamfer_union_list
from compas.datastructures import Mesh
import numpy as np
from compas.datastructures import Mesh
import compas_slicer.utilities as utils
import logging
import networkx as nx
from compas_slicer.slicers.slice_utilities import create_graph_from_mesh_vkeys
from compas_slicer.pre_processing.preprocessing_utils.geodesics import get_igl_EXACT_geodesic_distances, \
    get_custom_HEAT_geodesic_distances
import copy
import os
import json
from principal_directions import get_principal_directions,vector_projection,get_verctor_weigh
import statistics
from compas.datastructures import Mesh
from distances import dijkstra_distances,multi_source_dijkstra,multi_source_dijkstra_cannot_flip,not_flip_cube_distances_multi_sources
from distances import print_list_with_details

logger = logging.getLogger('logger')

class CompoundTargetDart(CompoundTarget):
    def __init__(self, mesh:Mesh, v_attr, value, DATA_PATH, union_method='min', union_params=[],
                 geodesics_method='exact_igl', anisotropic_scaling=False,Lowornot=False,notflip=False):
        self.low=Lowornot
        self.notflip=notflip
        super().__init__(mesh, v_attr, value, DATA_PATH, union_method=union_method, union_params=union_params,
                 geodesics_method=geodesics_method, anisotropic_scaling=anisotropic_scaling) 

        self.distances_from_max=[]
        self.distance_max=0
        self.distances_from_other=[0]*self.VN
        self.offset_distance_list=[0]*self.VN
        #print("self.offset_distance_list",self.offset_distance_list)
        self.offset_distance_weigh_list=[0]*self.VN
        self.pare_saddles=[0]*self.number_of_boundaries
        
        self.offset_distance_list_for_target=[]
    def get_saddles_number(self,saddles_n):
        self.saddles_number=saddles_n
    def set_other_target(self,other_target:'CompoundTargetDart'):
        self.other_target=other_target
        other_target.other_target=self
        
    def compute_geodesic_distances(self):
        """
        Computes the geodesic distances from each of the target's neighborhoods  to all the mesh vertices.
        Fills in the distances attributes.
        """
        if self.notflip:
            self.compute_not_flip_geodesic_distances
            return 0


        if self.low:
            outputname="distancess_Low.json"
        else:
            outputname="distancess_High.json"
        try:
            
            distances_lists= load_nested_list(file_path=self.OUTPUT_PATH,file_name=outputname)
            print("old distance")
        except:
            distances_lists =[get_igl_EXACT_geodesic_distances(self.mesh, vstarts) for vstarts in
                               self.clustered_vkeys]

            distances_lists = [list(dl) for dl in distances_lists]  # number_of_boundaries x #V

            save_nested_list(file_path=self.OUTPUT_PATH,file_name=outputname,nested_list=distances_lists)
        self.update_distances_lists(distances_lists)
    
    def compute_weigh_geodesic_distances(self):
        if self.low:
            outputname="distancess_cur_Low.json"
        else:
            outputname="distancess_cur_High.json"
        try:
            
            distances_lists= load_nested_list(file_path=self.OUTPUT_PATH,file_name=outputname)
            print("old distance")
        except:
            distances_lists = [multi_source_dijkstra(self.mesh, vstarts) for vstarts in
                               self.clustered_vkeys]

            distances_lists = [list(dl) for dl in distances_lists]  # number_of_boundaries x #V

            save_nested_list(file_path=self.OUTPUT_PATH,file_name=outputname,nested_list=distances_lists)
        self.update_distances_lists(distances_lists)


    def compute_not_flip_geodesic_distances(self,derection=True):
        if self.low:
            outputname="distancess_notflip_Low.json"
        else:
            outputname="distancess_notflip_High.json"
        try:
            
            distances_lists_nf= load_nested_list(file_path=self.OUTPUT_PATH,file_name=outputname)
            print("old distance")
        except:
            distances_lists_nf = [not_flip_cube_distances_multi_sources(self.mesh, vstarts,derection) for vstarts in
                               self.clustered_vkeys]

            distances_lists_nf = [list(dl) for dl in distances_lists_nf]  # number_of_boundaries x #V
            for i,xl in enumerate(distances_lists_nf):
                for j,x in enumerate(xl):
                    if not np.isinf(x):
                        distances_lists_nf[i][j]=self._distances_lists[i][j]

            save_nested_list(file_path=self.OUTPUT_PATH,file_name=outputname,nested_list=distances_lists_nf)
        self.update_distances_lists(distances_lists=distances_lists_nf,nf =True)



    def compute_edge_eigvecs(self):
        if self.low:
            outputname="weigh_cur_Low.json"
        else:
            outputname="weigh_cur_High.json"        
        try:
            edge_weigh= load_nested_list(file_path=self.OUTPUT_PATH,file_name=outputname)
            for i,(edge,data) in enumerate(self.mesh.edges(data=True)):
                 data['weigh_cur']=edge_weigh[i]
            print("old weigh of curviture")

        except:
            edge_weigh=[]
            print("new weigh of curviture")
            for vertex,data in self.mesh.vertices(data=True):
                 eigvecs =get_principal_directions(veky_center=vertex,mesh=self.mesh)
                 #print(eigvecs)
                 data['eigvecs_min']=eigvecs[0]
                 data['eigvecs_max']=eigvecs[1]
                 neighbors=self.mesh.vertex_neighbors(vertex)
                 self.compute_half_weigh(vertex1=vertex,vertex2s=neighbors)
            for edge,data in self.mesh.edges(data=True):
                print(edge,data['weigh_cur1'],data['weigh_cur2'])
                data['weigh_cur']=data['weigh_cur1']+data['weigh_cur2']
                edge_weigh.append(data['weigh_cur'])
            save_nested_list(file_path=self.OUTPUT_PATH,file_name=outputname,nested_list=edge_weigh)


    def compute_half_weigh(self,vertex1,vertex2s):
        for vertex2 in vertex2s:
            vector=np.array(self.mesh.vertex_coordinates(vertex1))-np.array(self.mesh.vertex_coordinates(vertex2))
            v1min=self.mesh.vertex_attribute(key=vertex1,name='eigvecs_min')
            v1max=self.mesh.vertex_attribute(key=vertex1,name='eigvecs_max')
            weigh1=get_verctor_weigh(vector,v1min,v1max)
            if vertex1<vertex2:
                #print("weigh_cur1",weigh1)
                data=self.mesh.edge_attributes(edge=(vertex1,vertex2))
                data['weigh_cur1']=(weigh1)*0.5
            else:
                #print("weigh_cur2",weigh1)
                data=self.mesh.edge_attributes(edge=(vertex1,vertex2))
                data['weigh_cur2']=(weigh1)*0.5
    
    def compute_curvature(self,vertex):
        # 1. 拟合局部平面
        a, b, c = self.fit_plane(vertex)

        # 2. 计算法向量 (实际上是给定的法向量)
        normal = np.array([a, b, -1])
        normal /= np.linalg.norm(normal)

        # 3. 计算梯度和 Hessian 矩阵
        gradient = np.array([a, b])
        Hessian = np.zeros((2, 2))

        # 4. 计算曲率张量
        norm_grad = np.linalg.norm(gradient)
        K = Hessian - np.outer(gradient, gradient) / norm_grad**2
        print("K",K)
        # 5. 求解曲率张量的特征值和特征向量
        eigvals, eigvecs = np.linalg.eigh(K)

        return eigvals, eigvecs
    
         
    def compute_uneven_boundaries_weight_max(self, other_target:'CompoundTarget'):
        """
        If the target has multiple neighborhoods/clusters of vertices, then it computes their maximum distance from
        the other_target. Based on that it calculates their weight_max for the interpolation process
        """
        if self.number_of_boundaries > 1:
            ds_avg_HIGH = self.get_boundaries_rel_dist_from_other_target(other_target)
            max_param = max_excluding_infinity(ds_avg_HIGH)
        
            print(len(ds_avg_HIGH),"ds_avg_HIGH",max_param)
            for i, d in enumerate(ds_avg_HIGH):  # offset all distances except the maximum one
                if abs(d - max_param) > 0.01:  # if it isn't the max value
                    ds_avg_HIGH[i] = d + self.offset

            self.weight_max_per_cluster = [d / max_param for d in ds_avg_HIGH]
            # this line is add by Yichuan
            self.distances_from_max=[max_param-d for d in ds_avg_HIGH]
            print("self.distances_from_max",self.distances_from_max)
            self.distance_max=max_param
            ### end here
            logger.info('weight_max_per_cluster : ' + str(self.weight_max_per_cluster))
        else:
            logger.info("Did not compute_norm_of_gradient uneven boundaries, target consists of single component")


    def union(self,list,union_method=None):
        if union_method==None:
            union_method=self.union_method
        union_params=self.union_params
        
        if union_method == 'min':
            # --- simple union
            return np.min(list)
        elif union_method == 'smooth':
            # --- blend (smooth) union
            return blend_union_list(values=list, r=union_params[0])
        elif union_method == 'chamfer':
            # --- blend (smooth) union
            return chamfer_union_list(values=list, r=union_params[0])
        elif union_method == 'stairs':
            # --- stairs union
            try:
                return stairs_union_list(values=list, r=union_params[0], n=union_params[1]) 
            except:
                print(self.union_params) 
                return blend_union_list(values=list, r=union_params[0])  


    def get_offset_distances_veky(self,veky):
        if len(self.offset_distance_list)>=1:
            list_distance=self.offset_distance_list
            #print("get offset diatance",list_distance[veky])
            return(list_distance[veky])
        else:
            self.get_distance
            return(self.get_distance(veky))
    

        
    def offset_distances(self,other_target:'CompoundTargetDart',get_offset_distances=False):
        if self.has_uneven_weights:
            if get_offset_distances:
                self.offset_distance_list_for_target=[]
                self.all_offset_distances_target=[]
                for veky in range(self.VN):
                    union_distance,all_distances=self.all_offset_distance(veky,other_target)
                    self.offset_distance_list_for_target.append(union_distance)
                    self.all_offset_distances_target.append(all_distances)
                print_list_with_details(self.all_offset_distances_target,"self.all_offset_distances_target")
            else:
                self.offset_distance_list_for_target=[]
                for veky in range(self.VN):
                    self.offset_distance_list_for_target.append(self.offset_distance(veky,other_target))
            print("Offset_distance_list_!!!!!!!!!!!!!!!!!!!!")
            
    

    def point_offset_distance_High(self,saddle):
        if self.mesh.vertex_normal(key=saddle)[2]<0:
            for veky in range(self.VN):
                self.offset_distance_list[veky]=self.get_distance(veky)
                #self.offset_distance_list[veky]=1000
        else:
            self.saddle_offset_distances(saddle)
            print("point_offset_distance_High")
        
    def point_offset_distance_Low(self,saddle):
        if self.mesh.vertex_normal(key=saddle)[2]>0:
            for veky in range(self.VN):
                self.offset_distance_list[veky]=self.get_distance(veky)
                #self.offset_distance_list[veky]=1000
        else:
            self.saddle_offset_distances(saddle) 
            print("point_offset_distance_Low") 
          
    def saddle_offset_distances(self,saddle):
        self.offset_distance_list_saddle=self.get_offset_distances_saddle(saddle)
        for veky in range(self.VN):
            self.offset_distance_list[veky]=self.saddle_offset_distance(veky,saddle)
                

    def saddle_offset_distance(self,veky,saddle):
        distance_list=copy.deepcopy(self.get_all_distances_for_vkey(veky))# list of floats (# number_of_boundaries)
        saddle_distance=self.offset_distance_list_saddle
        for i,distance in enumerate(distance_list):
            if saddle_distance[i]>0:
                distance_list[i]=distance+saddle_distance[i]
        if distance_list[0]==distance_list[1]:
            print("offset distance",veky,distance_list,)
        return self.union(distance_list)
    
    def get_offset_distances_saddle_with_not_flip_requsted(self,saddle,sides_limitation=None):
        saddle_distances_from_boundary_list=[0]*self.number_of_boundaries
        for target_number in range(self.number_of_boundaries):
            distance=self.get_all_distances_for_vkey(saddle,True)[target_number]
            neibors=self.mesh.vertex_neighborhood(key=saddle,ring=6)

            if np.isinf(distance):
                for neibor in neibors:
                    distance_nei=self.get_all_distances_for_vkey(neibor,True)[target_number]
                    if not np.isinf(distance_nei):
                        print("saddle is using neibor not flip distance !!!!!!!")
                        distance=self.get_all_distances_for_vkey(saddle,False)[target_number]
                        break
                    
            
            saddle_distances_from_boundary_list[target_number]=(distance)
        if sides_limitation is not None:
            retain_smallest_x_non_none(saddle_distances_from_boundary_list,sides_limitation)    
        
        try:

            max_distance = max(value for value in saddle_distances_from_boundary_list if value != float('inf') and value != -float('inf') and value is not None)
            for i,distance in enumerate(saddle_distances_from_boundary_list):
                if distance != float('inf') and distance != -float('inf') and distance is not None:
            
                    saddle_distances_from_boundary_list[i]=max_distance-distance  
                else:
                    saddle_distances_from_boundary_list[i]=None
        except:
            print("get_offset_distances_saddle error")
            max_distance=max(saddle_distances_from_boundary_list)
            for i,distance in enumerate(saddle_distances_from_boundary_list):
                saddle_distances_from_boundary_list[i]=max_distance-distance
        
            
           
        print("saddle_distances_from_boundary_list",saddle_distances_from_boundary_list,saddle)
        return(saddle_distances_from_boundary_list)
    
    def get_offset_distances_saddle(self,saddle,sides_limitation=None):
        saddle_distances_from_boundary_list=[0]*self.number_of_boundaries
        for target_number in range(self.number_of_boundaries):
            distance=self.get_all_distances_for_vkey(saddle,False)[target_number]
            neibors=self.mesh.vertex_neighborhood(key=saddle,ring=6)

            if np.isinf(distance):
                for neibor in neibors:
                    distance_nei=self.get_all_distances_for_vkey(neibor,False)[target_number]
                    if not np.isinf(distance_nei):
                        print("saddle is using neibor not flip distance !!!!!!!")
                        distance=self.get_all_distances_for_vkey(saddle,False)[target_number]
                        break
                    
            
            saddle_distances_from_boundary_list[target_number]=(distance)
        if sides_limitation is not None:
            retain_smallest_x_non_none(saddle_distances_from_boundary_list,sides_limitation)    
        
        try:

            max_distance = max(value for value in saddle_distances_from_boundary_list if value != float('inf') and value != -float('inf') and value is not None)
            for i,distance in enumerate(saddle_distances_from_boundary_list):
                if distance != float('inf') and distance != -float('inf') and distance is not None:
            
                    saddle_distances_from_boundary_list[i]=max_distance-distance  
                else:
                    saddle_distances_from_boundary_list[i]=None
        except:
            print("get_offset_distances_saddle error")
            max_distance=max(saddle_distances_from_boundary_list)
            for i,distance in enumerate(saddle_distances_from_boundary_list):
                saddle_distances_from_boundary_list[i]=max_distance-distance
        
            
           
        print("saddle_distances_from_boundary_list",saddle_distances_from_boundary_list,saddle)
        return(saddle_distances_from_boundary_list)
    def get_all_distances_for_vkey(self, i,nf=False):
        """ Returns distances from each cluster separately for vertex i. Smooth union doesn't play here any role. """
        if nf:
            return [self._distances_lists_nf[list_index][i] for list_index in range(self.number_of_boundaries)]
        else:
            return [self._distances_lists[list_index][i] for list_index in range(self.number_of_boundaries)]



            


    
    def offset_distance(self,veky,other_target:'CompoundTargetDart'):
        """
        对于每个target的内部的每条线 都移动到距离对面target相同距离的位置
        """
        distance_list=(self.get_all_distances_for_vkey(veky))# list of floats (# number_of_boundaries)
        offset_distance_list=[0]*len(distance_list)
        other_distance=other_target.get_offset_distances_veky(veky)
        for i,distance in enumerate(distance_list):
            if self.distances_from_max[i]>0:
                offset_distance_list[i]=distance+(self.distances_from_max[i]+200)*((distance+other_distance)/self.distance_max)*(self.scale)
                #print((self.distances_from_max[i]+200)*((distance+other_distance)/self.distance_max)*(self.scale),distance,offset_distance_list[i],i)
            else:
                offset_distance_list[i]=distance
        #print("offset distance",veky,offset_distance_list)
        return self.union(offset_distance_list)
    
    def all_offset_distance(self,veky,other_target:'CompoundTargetDart'):
        """
        对于每个target的内部的每条线 都移动到距离对面target相同距离的位置
        """
        distance_list=(self.get_all_distances_for_vkey(veky))# list of floats (# number_of_boundaries)
        diff_offset_distance_list=[0]*len(distance_list)
        offset_distance_list=[0]*len(distance_list)
        other_distance=other_target.get_offset_distances_veky(veky)
        for i,distance in enumerate(distance_list):
            if self.distances_from_max[i]>0:
                diff_offset_distance_list[i]=(self.distances_from_max[i]+200)*((distance+other_distance)/self.distance_max)*(self.scale)
                offset_distance_list[i]=distance+diff_offset_distance_list[i]
                
            else:
                offset_distance_list[i]=distance
        #print("all_offset distance",veky,diff_offset_distance_list)
        union_distance=self.union(offset_distance_list)
        return union_distance,diff_offset_distance_list
    def caculate_saddle_distance(self,saddle):
        pass
    ###end
    def update_distances_lists(self, distances_lists,nf=False):
        """
        Fills in the distances attributes.
        """
        if not nf:
            self._distances_lists = distances_lists
            self._distances_lists_flipped = []  # empty
            for i in range(self.VN):
                current_values = [self._distances_lists[list_index][i] for list_index in range(self.number_of_boundaries)]
                self._distances_lists_flipped.append(current_values)
            self._np_distances_lists_flipped = np.array(self._distances_lists_flipped)
            self._max_dist = np.max(self._np_distances_lists_flipped)
        else:
            self._distances_lists_nf = distances_lists
            self._distances_lists_flipped_nf = []  # empty
            for i in range(self.VN):
                current_values = [self._distances_lists_nf[list_index][i] for list_index in range(self.number_of_boundaries)]
                self._distances_lists_flipped_nf.append(current_values)
            self._np_distances_lists_flipped_nf = np.array(self._distances_lists_flipped_nf)
            # self._max_dist = np.max(self._np_distances_lists_flipped)            
    
    # def offset_distance_with_saddles_and_targets(self,saddle_offset_distances_list,nameHL=None,basic_offset=0):
    #     try:
    #         print("offset_distance_with_saddles_and_targets"+nameHL)
    #     except:
    #         print("offset_distance_with_saddles_and_targets")
    #     for veky in range(self.VN):#every point
    #         self.offset_distance_with_saddles_and_targets_veky(saddle_offset_distances_list,veky,nameHL,basic_offset)

    # def offset_distance_with_saddles_and_targets_veky(self,saddle_offset_distances_list,veky,nameHL=None,basic_offset=0):
    #     if nameHL==None:
    #         name='influences'
    #     else:
    #         name=nameHL+'_influences'
    #     influence_list=self.mesh.vertex_attribute(key=veky,name=name)
    #     org_distances=self.get_all_distances_for_vkey(veky)
    #     offset_distances=copy.deepcopy(org_distances)
    #     #print(veky,influence_list)
    #     for i,offset_list in enumerate(saddle_offset_distances_list):#every saddle
    #         influence=influence_list[i]           
    #         for j,offset in enumerate(offset_list): #every target
    #             if influence>0:
    #                 offset=influence*offset
    #                 offset_distances[j]+=offset
    #             #offset_distances[j]+=basic_offset*(1-sum(offset_list))
        
    #     # print(veky)
    #     print(len(self.offset_distance_list),veky)
    #     #self.offset_distance_list.append(0)
    #     self.offset_distance_list[veky]=self.union(offset_distances)
    
    def offset_distance_with_saddles_targets(self,saddles_targets_offset_distances_list,other_target_self_offset_distance_list,basic_offset=0,frame='',single_side=False):
        """
        对于所有的顶点计算到本目标的综合距离，基于所有鞍点和边缘的影响场
        """
        self.offset_distance_list=[0]*self.VN
        if self.low:
            print("offset_distance_with_saddles"+"_Low")
        else:
            print("offset_distance_with_saddles"+"_High")
        # 先要得到一个临时的对向距离，用于平滑边界
        self.get_temporary_distance_list_other_side(other_target_self_offset_distance_list)
        nested_distances=[]
        for veky in range(self.VN):#every point
            distances=self.offset_distance_with_saddles_targets_veky(saddles_targets_offset_distances_list,veky,basic_offset,single_side=single_side)
            nested_distances.append(distances)
        if self.low:
            ifHL='_low_'
            save_nested_list(file_path=self.OUTPUT_PATH,file_name="offset_distances"+ifHL+str(frame)+'.json',nested_list=nested_distances)
        # print_list_with_details(self.offset_distance_list,"self.offset_distance_list")


                 
    def get_temporary_distance_list_other_side(self,other_target_self_offset_distance_list):
        """
        用自己边界的影响场参数计算对面的目标距离，对于所有的点
        """
        
        self.temporary_distance_list_other_side=[0]*self.VN
        for veky in range(self.VN):
            self.temporary_distance_list_other_side[veky]=self.get_temporary_distance_other_side(veky,other_target_self_offset_distance_list)
        self.avg_temporary_distances_other_side = []
        for i in range(self.number_of_boundaries):
            sum_value = 0
            for j in range(len(self.clustered_vkeys[i])):
                sum_value += self.temporary_distance_list_other_side[self.clustered_vkeys[i][j]]
            average_value = sum_value / len(self.clustered_vkeys[i])
            self.avg_temporary_distances_other_side.append(average_value)
        print_list_with_details(self.temporary_distance_list_other_side,"temporary_distance_list_other_side")
        print(self.avg_temporary_distances_other_side,"target value")

    def get_temporary_distance_other_side(self,veky,other_target_self_offset_distance_list):
        """
        求出自己附近到对面距离 只靠路自己的边界的影响场的偏移影响 对于一个点
        """
        name='influences'
        influence_list=self.mesh.vertex_attribute(key=veky,name=name)
        other_offset_distances=self.other_target.get_all_distances_for_vkey(veky)
        for i,offset_list in enumerate(other_target_self_offset_distance_list):#self target from other target
            if not self.low:
                i_needed=list(range(self.saddles_number,self.saddles_number+self.number_of_boundaries))
            else:
                i_needed=list(range(self.saddles_number+self.other_target.number_of_boundaries,self.saddles_number+self.other_target.number_of_boundaries+self.number_of_boundaries))
            if i in i_needed:
                influence=influence_list[i]  
                for j,offset in enumerate(offset_list):
                    offset=influence*offset
                    other_offset_distances[j]+=offset
        return self.union(other_offset_distances)



            
    def offset_distance_with_saddles_targets_veky(self,saddles_targets_offset_distances_list,veky,basic_offset=0,single_side=False):
        """
        求单个顶点的综合距离，针对list中有none的部分应该用别的来填补
        """
 
        name='influences'

        saddle_offset_distances_list=saddles_targets_offset_distances_list[0]
        target_self_offset_distance_list=saddles_targets_offset_distances_list[1]
        target_other_offset_distance_list=saddles_targets_offset_distances_list[2]

        
        influence_list=self.mesh.vertex_attribute(key=veky,name=name)
        offset_distances=[0]*self.number_of_boundaries

        none_influence=[0]*self.number_of_boundaries
        none_influence_distance=[0]*self.number_of_boundaries
        #print(len(offset_distances),len(saddle_offset_distances_list[0]),len( target_self_offset_distance_list[0]),len(target_other_offset_distance_list[0]),len(list(self.get_all_distances_for_vkey(veky))))
        #把每个鞍点的偏移距离乘以影响加上
        for i,offset_list in enumerate(saddle_offset_distances_list):#every saddle
            influence=influence_list[i]           
            for j,offset in enumerate(offset_list): #every target
                if influence>0:
                    if isinstance(offset,list):
                        none_influence[j]+=influence
                        none_influence_distance[j]+=offset[1]
                    else:
                        offset=influence*offset
                        offset_distances[j]+=offset
        #把对面边界的影响下的偏移距离加上（主要影响对象边界附近）
        for i,offset_list in enumerate(target_other_offset_distance_list):#other target
            if not self.low:#high target_other target is low target
                influence=influence_list[self.saddles_number+self.other_target.number_of_boundaries+i]
            else:# low target_other target is high target
                influence=influence_list[self.saddles_number+i]
            for j,offset in enumerate(offset_list): #every boundary of this targets
                if influence>0:
                    if isinstance(offset,list):
                        none_influence[j]+=influence
                        none_influence_distance[j]+=offset[1]
                    else:
                        offset=influence*offset
                        
                        offset_distances[j]+=offset
        if single_side:
            #论文示范参差边界切片
            for i,offset_list in enumerate(target_self_offset_distance_list):#other target
                if not self.low:#high target_other target is low target
                    influence=influence_list[self.saddles_number+i]
                else:# low target_other target is high target
                    influence=influence_list[self.saddles_number+self.other_target.number_of_boundaries+i]

                for j,offset in enumerate(offset_list): #every boundary of this targets
                    if influence>0:
                        if isinstance(offset,list):
                            none_influence[j]+=influence
                            none_influence_distance[j]+=offset[1]
                        else:

                            offset=influence*offset
                            offset_distances[j]+=offset        
        else:
            #针对本边界对于偏移距离需要做出调整    
            for i,offset_list in enumerate(target_self_offset_distance_list):#this target with i as boundary numbers
                # 判断是高边界还是低边界
                if not self.low:#high target
                    influence=influence_list[self.saddles_number+i]
                else:# low target
                    influence=influence_list[self.saddles_number+self.other_target.number_of_boundaries+i]

                for j,offset in enumerate(offset_list):
                    if influence>0:
                        if isinstance(offset,list):
                            none_influence[j]+=influence
                            none_influence_distance[j]+=offset[1]
                        else:
                            if self.temporary_distance_list_other_side[veky]==0:
                                offset=offset*influence
                            else:  
                                offset=(self.temporary_distance_list_other_side[veky])*offset*influence/self.avg_temporary_distances_other_side[i]
                                #print(veky,i,j,self.temporary_distance_list_other_side[veky],self.get_distance(veky),((self.temporary_distance_list_other_side[veky]+self.get_distance(veky))/self.avg_temporary_distances_other_side[i]))
                                offset_distances[j]+=offset

        org_distances=self.get_all_distances_for_vkey(veky)
        for j,offset_dist in enumerate(offset_distances):
            if none_influence[j]<1:
                offset_distances[j]=offset_dist/(1-none_influence[j])
            offset_distances[j]+=org_distances[j]
            #offset_distances[j]+=none_influence_distance[j]

        #print(len(self.offset_distance_list))
        if self.low:
            method='min'
        else:
            method='min'
        self.offset_distance_list[veky]=self.union(offset_distances,method)
        if veky==0:
            print(veky, self.offset_distance_list[veky])
        return offset_distances
    def offset_distance_with_saddles(self,saddle_offset_distances_list,nameHL=None,basic_offset=0):
        try:
            print("offset_distance_with_saddles"+nameHL)
        except:
            print("offset_distance_with_saddles")
        for veky in range(self.VN):#every point
            
            self.offset_distance_with_saddles_veky(saddle_offset_distances_list,veky,nameHL,basic_offset)

    
    def smooth_boundary_distances(self,other_target:'CompoundTargetDart',ring=5):
        """
        smooth bondary after saddle offsset
        """
        for i,vstarts in enumerate(self.clustered_vkeys):
            print(i,"smooth target")
            neibours_set=set()
            ecozoro=False
            for veky in vstarts:
                if self.offset_distance_list[veky]==0:
                    ecozoro=True
                    break
            if ecozoro:
                for veky in vstarts:
                    self.check_point_near_boundarys(neibours_set=neibours_set,veky=veky,ring=ring)
                    self.offset_distance_list[veky]=0               
            disdediss=[]
            for veky in vstarts:
                self.check_point_near_boundarys(neibours_set=neibours_set,veky=veky,ring=ring)
                distance1=self.offset_distance_list[veky]
                distance2=other_target.offset_distance_list[veky]
                disdedis=distance1/distance2
                disdediss.append(disdedis)
            avedisdedis=np.mean(disdediss)
            for veky in vstarts:
                distance2=other_target.offset_distance_list[veky]
                distance1=avedisdedis*distance2
                self.offset_distance_list[veky]=distance1
            neibours_set-=set(vstarts)
            mesh_smooth_centroid_vertices_offset_distance(mesh=self.mesh,unfixed=list(neibours_set),kmax=300,damping=0.5,offset_distance_list=self.offset_distance_list)


    def check_point_near_boundarys(self,neibours_set:set,veky,ring=3):
        neibours=self.mesh.vertex_neighborhood(key=veky,ring=ring)
        neibours_set.update(neibours)            
    def offset_distance_with_saddles_veky(self,saddle_offset_distances_list,veky,nameHL=None,basic_offset=0):
        if nameHL==None:
            name='influences'
        else:
            name=nameHL+'_influences'
        influence_list=self.mesh.vertex_attribute(key=veky,name=name)
        org_distances=self.get_all_distances_for_vkey(veky)
        offset_distances=copy.deepcopy(org_distances)
        #print(veky,influence_list)
        for i,offset_list in enumerate(saddle_offset_distances_list):#every saddle
            influence=influence_list[i]           
            for j,offset in enumerate(offset_list): #every target
                if influence>0:
                    offset=influence*offset
                    offset_distances[j]+=offset
                # print(sum(offset_list))
                # offset_distances[j]+=basic_offset*(1-sum(offset_list))
        # print(veky)
        # print(len(self.offset_distance_list))
        self.offset_distance_list.append(0)
        self.offset_distance_list[veky]=self.union(offset_distances)

    def save_offset_distance(self,name):
        """
        Yichuan Save the offseted distance to json
        """
        utils.save_to_json(self.offset_distance_list, self.OUTPUT_PATH, name)
    
    def load_offset_distance(self,name):
        """
        Yichuan Save the offseted distance to json
        """
        self.offset_distance_list=load_nested_list(file_path=self.OUTPUT_PATH,file_name=name)
        print('load offset distance',len(self.offset_distance_list),len(list(self.mesh.vertices())))

def max_excluding_infinity(numbers):
    # 过滤掉无穷大
    finite_numbers = [num for num in numbers if num != float('inf')]
    
    # 如果所有数都是无穷大，返回 None 或适当的值
    if not finite_numbers:
        return None
    
    # 返回过滤后的最大值
    return max(finite_numbers)

def save_nested_list(file_path, file_name, nested_list):
    """
    
    
    :param file_path: 
    :param file_name:
    :param nested_list:
    """
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    full_path = os.path.join(file_path, file_name)
    
    
    with open(full_path, 'w') as file:
        json.dump(nested_list, file)
    print("save_nested_list_",file_name)

def load_nested_list(file_path, file_name):

    full_path = os.path.join(file_path, file_name)
    with open(full_path, 'r') as file:
        nested_list = json.load(file)
    print("load_nested_list_",file_name)
    return nested_list

def mesh_smooth_centroid_vertices_offset_distance(mesh:Mesh, unfixed, kmax=10, damping=0.5, callback=None, callback_args=None,offset_distance_list=None):
    """Smooth a mesh by moving every free vertex to the centroid of its neighbors.

    Parameters
    ----------
    mesh : :class:`~compas.datastructures.Mesh`
        A mesh object.
    fixed : list[int], optional
        The fixed vertices of the mesh.
    kmax : int, optional
        The maximum number of iterations.
    damping : float, optional
        The damping factor.
    callback : callable, optional
        A user-defined callback function to be executed after every iteration.
    callback_args : list[Any], optional
        A list of arguments to be passed to the callback.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If a callback is provided, but it is not callable.

    """
    if callback:
        if not callable(callback):
            raise Exception("Callback is not callable.")


    for k in range(kmax):

        for key in unfixed:
            if key in unfixed:
                x  = offset_distance_list[key]
                neibordistance=([mesh.edge_length(key,nbr) for nbr in mesh.vertex_neighbors(key)])
                neiborweigh=[1/(dist) for dist in neibordistance]
                sumneiborweigh=sum(neiborweigh)
                de_sum=1/sumneiborweigh
                neiborweigh=[x*de_sum for x in neiborweigh]

                cx=sum([offset_distance_list[nbr]*neiborweigh[i] for i,nbr in enumerate(mesh.vertex_neighbors(key))])

                offset_distance_list[key] += damping * (cx - x)


        if callback:
            callback(k, callback_args)

def mesh_smooth_centroid_vertices_influence(mesh:Mesh, fixed, kmax=10, damping=0.5, callback=None, callback_args=None,influence_list=None):
    """Smooth a mesh by moving every free vertex to the centroid of its neighbors.

    Parameters
    ----------
    mesh : :class:`~compas.datastructures.Mesh`
        A mesh object.
    fixed : list[int], optional
        The fixed vertices of the mesh.
    kmax : int, optional
        The maximum number of iterations.
    damping : float, optional
        The damping factor.
    callback : callable, optional
        A user-defined callback function to be executed after every iteration.
    callback_args : list[Any], optional
        A list of arguments to be passed to the callback.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If a callback is provided, but it is not callable.

    """
    if callback:
        if not callable(callback):
            raise Exception("Callback is not callable.")

    len_list_in=len(influence_list[0])
    for k in range(kmax):

        for key in mesh.vertices():
            if key not in fixed:
                x  = influence_list[key]
                neibordistance=([mesh.edge_length(key,nbr) for nbr in mesh.vertex_neighbors(key)])
                neiborweighs=[1/(dist) for dist in neibordistance]
                sumneiborweigh=sum(neiborweighs)
                de_sum=1/sumneiborweigh
                neiborweighs=[neiborweigh*de_sum for neiborweigh in neiborweighs]

                # 初始化新的影响力数组
                new_influence = [0] * len_list_in

                for i in range(len_list_in):
                    cx = sum([influence_list[nbr][i] * neiborweighs[j] for j, nbr in enumerate(mesh.vertex_neighbors(key))])
                    new_influence[i] = x[i] + damping * (cx - x[i])

                influence_list[key] = new_influence



        if callback:
            callback(k, callback_args)
def retain_smallest_x_non_none(data_list, x):
    # 获取非None元素及其索引
    non_none_items = [(item, index) for index, item in enumerate(data_list) if item is not None]
    
    # 如果非None元素的数量少于或等于x，则不需要做任何事情
    if len(non_none_items) <= x:
        print("No need to retain smallest x non-None elements.")
        return data_list
    
    # 对非None元素按值排序并获取最小的x个元素的索引
    smallest_x_indices = [index for value, index in sorted(non_none_items, key=lambda p: (p[0] is not None, p[0]))[:x]]
    


    
    # 保留最小的x个非None元素
    for index,x in enumerate(data_list):
        if index not in smallest_x_indices:
            data_list[index] = 0
        else:

            smallest_x_indices.remove(index)
    print(data_list,'______________')    
    return data_list
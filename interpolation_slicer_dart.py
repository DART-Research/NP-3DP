import numpy as np
import rdp as rdp
from compas_slicer.slicers import BaseSlicer
import logging
import progressbar
from compas_slicer.parameters import get_param
#from compas_slicer.pre_processing import assign_interpolation_distance_to_mesh_vertices
from compas_slicer.slicers.slice_utilities import ScalarFieldContours
from compas_slicer.geometry import VerticalLayersManager
#this line is add by YIchuan
from contour_dart import ScalarFieldContours_layer
from layer_dart import VerticalLayer_dart,path_dart
from compas.datastructures import Mesh,trimesh_split_edge
from compas_slicer.slicers import InterpolationSlicer
from interpolationdart import DartPreprocesssor
from layer_dart import verticalLayerManager_dart
import numpy as np
import logging
import progressbar
from compas.geometry import Point
import compas_slicer.utilities as utils
from compas.plugins import PluginNotInstalledError
from compas.geometry import distance_point_point
from compas.geometry import Vector
from compas.geometry import distance_point_point_sqrd
import compas_slicer
from seams_align_dart import seams_align
import copy
from unify_paths_orientation_dart import unify_paths_orientation


logger = logging.getLogger('logger')

__all__ = ['InterpolationSlicer_Dart','simplify_paths_rdp_with_gr','seams_smooth_with_gr']

class InterpolationSlicer_Dart(InterpolationSlicer):
    def __init__(self, mesh: Mesh, preprocessor:DartPreprocesssor=None, parameters=None,slice_on_boundary=True,gradient_evaluation=None,outputpath=None):
        super().__init__(mesh, preprocessor, parameters)
        self.preprocessor = preprocessor
        self.ifboundary_layer=[]
        self.slice_on_boundary=slice_on_boundary
        self.gradient_evaluation=gradient_evaluation
        self.outputpath=outputpath

    def get_scalar_field_without_weigh(self):
        for vkey in self.mesh.vertices():
            self.mesh.vertex[vkey]['scalar_field_org']=self.mesh.vertex[vkey]['scalar_field']
        

    def generate_paths(self,n=0,params_list=None):
        """ Generates curved paths. """
        if params_list is None:
            print('no params_list given, will generate automatically',params_list)
            if n==0:

                avg_layer_height = get_param(self.parameters, key='avg_layer_height', defaults_type='layers')
                n = find_no_of_isocurves(self.preprocessor.target_LOW, self.preprocessor.target_HIGH, avg_layer_height)

            if self.slice_on_boundary:
                target_params_list,target_params_L_H=get_targets_field(self.mesh)
                params_list,self.ifboundary_layer = get_interpolation_parameters_list_with_targets_value(n,target_params_list,target_params_L_H)
            else:
                params_list=get_interpolation_parameters_list(n)
        else:
            n=len(params_list)
        logger.info('%d paths will be generated' % n)

        
        
        #self.get_scalar_field_without_weigh()
        save_scalar_field_org(self.mesh)
        self.generate_paths_with_weights(params_list)
        print('slicing params list',params_list)
        reset_scalar_field(self.mesh)
        

    def generate_paths_with_weights(self,params_list,output_edge=False,saddles=None,face_edit_data=None,edge_edit_data=None):
        
        # create paths + layers
        max_dist = get_param(self.parameters, key='vertical_layers_max_centroid_dist', defaults_type='layers')
        vertical_layers_manager = verticalLayerManager_dart(max_dist)
        with progressbar.ProgressBar(max_value=len(params_list)) as bar:
            edges_needed=set(self.mesh.edges())
            discard_layer_num=0
            for i, param in enumerate(params_list):
                #print('layer',i,param)
                #assign_interpolation_distance_to_mesh_vertices(self.mesh, param, self.preprocessor.target_LOW,self.preprocessor.target_HIGH)
                change_scalar_field_by_weigh(self.mesh, param)
 
                contours = ScalarFieldContours_layer(self.mesh,i,self.gradient_evaluation,edges_needed,discard_layer_num,output_edge)
                #contours = ScalarFieldContoursweighed(self.mesh,param)
                contours.compute()

                edges_needed,discard_layer_num=contours.add_to_vertical_layers_manager(vertical_layers_manager,i,output_edge=output_edge)
                if saddles is not None:
                    saddle=saddles[i]
                    saddle_neibour_edges=self.mesh.vertex_edges(saddle)
                    layer=vertical_layers_manager.layers[i]
                    saddle_paths=layer.get_path()
                    #print('path number',len(saddle_paths))
                    for saddle_path in saddle_paths:
                        path_edges=saddle_path.get_edges()
                        if not List_has_intersection(saddle_neibour_edges,path_edges):
                            layer.remove_path(saddle_path)
                    #print('path number after',len(saddle_paths))
                    slice_mesh(layer=layer,edges_cutting_data=edge_edit_data,faces_cutting_data=face_edit_data,mesh=self.mesh,weight=param,edges_needed=edges_needed)
                    edges_needed=set(self.mesh.edges())
                bar.update(i)  # advance progress bar

        self.layers = vertical_layers_manager.layers
    
    def get_layer(self, layer_index)->VerticalLayer_dart:
        return self.layers[layer_index]
    def remove_layers(self,layer_ids):
        indices_to_remove = sorted(layer_ids, reverse=True)
        for index in indices_to_remove:
            if index < len(self.layers):
                del self.layers[index]
    

    @property
    def get_layer_number(self)->int:
        return  len(self.layers)
        
    def close_paths(self):
        """ For paths that are labeled as closed, it makes sure that the first and the last point are identical. """
        for layer in self.layers:
            for path in layer.paths:
                if path.is_closed:  # if the path is closed, first and last point should be the same.
                    if distance_point_point_sqrd(path.points[0]['point'], path.points[-1]['point']) > 0.00001:  # if not already the same
                        path.points.append(path.points[0])

    def post_processing(self):
        """Applies standard post-processing operations: seams_align and unify_paths."""
        self.close_paths()

        #  --- Align the seams between layers and unify orientation
        seams_align(self, align_with='next_path')
        unify_paths_orientation(self)

        self.close_paths()
        logger.info("Created %d Layers with %d total number of points" % (len(self.layers), self.number_of_points))

    def save_slicer(self,outputpath,name):
        data=[]
        for layer_id in range(self.number_of_layers):
            layer=self.get_layer(layer_id)
            data.append([])
            for path in layer.get_path():
                path_data = path.to_data(True)
               
                data[-1].append(path_data)
        utils.save_to_json(data=data,filepath=outputpath,name= name)
    @classmethod
    def load_slicer_from_file(cls,outputpath,name):
        cls.layers=[]
        data=utils.load_from_json(filepath=outputpath,name= name)
        for layer_id,layer_data in enumerate(data):
         
            cls.layers.append(VerticalLayer_dart(id=layer_id))
            layer=cls.layers[layer_id]
            for path_id,path_data in enumerate(layer_data):
                path=path_dart.from_dict(data=path_data)
                layer.append_(path)
    def slice_model(self, weights_list=None):
        """Slices the model and applies standard post-processing and removing of invalid paths."""

        self.generate_paths(params_list=weights_list)
        self.remove_invalid_paths_and_layers()
        self.post_processing()

    


def change_scalar_field_by_weigh(mesh:Mesh,weigh):
    for vkey in mesh.vertex:
        
        d=mesh.vertex_attribute(vkey,'scalar_field_org')
  
       
        
        mesh.vertex[vkey]['scalar_field']=d-weigh
      

        
def save_scalar_field_org(mesh:Mesh):
    for vkey in mesh.vertex:
        
        d=mesh.vertex_attribute(vkey,'scalar_field')
        #print("save org",vkey)
        mesh.vertex[vkey]['scalar_field_org']=d 
def reset_scalar_field(mesh:Mesh):
    for vkey in mesh.vertex:
        
        d=mesh.vertex_attribute(vkey,'scalar_field_org')
        #print("save org",vkey)
        mesh.vertex[vkey]['scalar_field']=d     

#The below code is copy from compas_slicer.slicers
def find_no_of_isocurves(target_0, target_1, avg_layer_height=1.1):
    """ Returns the average number of isocurves that can cover the get_distance from target_0 to target_1. """
    avg_ds0 = target_0.get_avg_distances_from_other_target(target_1)
    avg_ds1 = target_1.get_avg_distances_from_other_target(target_0)
    number_of_curves = ((avg_ds0 + avg_ds1) * 0.5) / avg_layer_height
    return max(1, int(number_of_curves))


def get_interpolation_parameters_list(number_of_curves):
    """ Returns a list of #number_of_curves floats from 0.001 to 0.997. """
    # t_list = [0.001]
    t_list = []
    a = list(np.arange(number_of_curves + 1) / (number_of_curves + 1))
    a.pop(0)
    t_list.extend(a)
    t_list.append(1)
    t_list=[x-0.003 for x in t_list]
    
    return t_list
     
def get_interpolation_parameters_list_with_targets_value(number_of_curves,targets_value,target_L_H):
    """ Returns a list of #number_of_curves floats from 0.001 to 0.997. 
    every target curve's scalar field's value will be in the outfut list
    value are devides uniformly between target curve's scalar field values'
    
    number_of_curves: int
    targets_value: list of floats between 0 and 1 must be sroted and start with 0 and end with 1
    target_L_H: list of int of 1 or 2, must start with 1 and end with 2
    """
    a = list(np.arange(number_of_curves + 1) / (number_of_curves + 1))
    min_interval=1/(number_of_curves + 1)
    layer_loacation=[target_L_H[0]]
    # 结果列表，初始化为原列表的第一个元素  
    result = [targets_value[0]]  
      
    # 遍历列表中的每个元素（从第二个元素开始）  
    for i in range(1, len(targets_value)):  
        # 获取当前元素和前一个元素  
        current = targets_value[i]  
        current_L_H = target_L_H[i]
        prev = targets_value[i-1]  
          
        # 计算区间长度  
        interval_length = current - prev  
          
        # 计算理论上可以插入的最大数量（减一是因为要包含前一个元素）  
        # 但由于我们想要至少保持min_interval的间隔，所以可能需要调整  
        max_inserts = max(1, int(interval_length //min_interval)+1)  
          
        # 如果区间长度小于min_interval，则不插入  
        if max_inserts == 1:  
            result.append(current) 
            if current_L_H == 1:
                layer_loacation.append(current_L_H)
            else:    
                layer_loacation.append(0)
                layer_loacation[i-1]=current_L_H
            continue  
          
        # 计算实际的间隔  
        actual_interval = interval_length / max_inserts  
          
        # 插入值，插入值0到layer_loacation  
        for j in range(1, max_inserts):  
            # 插入值 = 前一个元素 + (j * 实际间隔)  
            insert_value = prev + (j * actual_interval)  
            result.append(insert_value)
            layer_loacation.append(0)
          
        # 添加当前元素到结果列表中  
        result.append(current)  
        if current_L_H == 1:
            layer_loacation.append(current_L_H)
        else:    
            layer_loacation.append(0)
            layer_loacation[i-1]=current_L_H

    result.remove(targets_value[0])  
    layer_loacation.pop(-1) 
    result=[i-0.003 for i in result]
    print("out fiels values",result) 
    return result,layer_loacation

def get_targets_field(mesh:Mesh):
    """   
    Returns two lists one of scalar fields for lower and higher boundary curves
    the other list indicating the lower and higher of each value.
    """
    field_list1=[]
    field_list2=[]

    for i in mesh.vertices():
        boundary=mesh.vertex_attribute(key=i,name='boundary')
        if boundary==1 :
            field=mesh.vertex_attribute(key=i,name='scalar_field')
            add_data_with_tolerance(field_list1,field,0.02)
        elif boundary==2:
            field=mesh.vertex_attribute(key=i,name='scalar_field')
            add_data_with_tolerance(field_list2,field,0.02)
    field_list,field_location=merge_and_track(field_list1,field_list2)
    print("lower boundary curve's scalar field",field_list1)
    print("higher boundary curve's scalar field",field_list2)
    return field_list,field_location

def add_data_with_tolerance(data_list, new_data, tolerance): 
    """
    if new data has no data closer than tolerance in data list than add new data to data list
    """ 
    # 标记是否需要添加新数据  
    should_add = True  
    # 遍历列表中的每个元素  
    for index, existing_data in enumerate(data_list):  
        # 计算差距  
        difference = abs(new_data - existing_data)  
        # 如果差距小于或等于误差  
        if difference <= tolerance:  
            # 更新该元素为两个值中的较大者  
            data_list[index] = max(existing_data, new_data)  
            # 标记不需要再添加新数据  
            should_add = False  
            break  # 如果只保留一个值，可以在这里中断循环
    # 如果应该添加新数据，则添加  
    if should_add:  
        data_list.append(new_data) 

def merge_and_track(list_a, list_b):
    """ 
    Returns a merged list and a list indicating the origin of each value.
    """
    # 使用元组 (value, origin) 的形式来存储值及其来源
    combined_list = [(val, 1) for val in list_a] + [(val, 2) for val in list_b]
    
    # 对合并后的列表按照值进行排序
    sorted_list = sorted(combined_list, key=lambda x: x[0])
    
    # 分离排序后的值和其对应的来源
    sorted_values = [item[0] for item in sorted_list]
    origins = [item[1] for item in sorted_list]
    
    return sorted_values, origins
def simplify_paths_rdp_with_gr(slicer:InterpolationSlicer_Dart, threshold):
    """Simplifies a path using the Ramer–Douglas–Peucker algorithm, implemented in the rdp python library.
    https://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm

    Parameters
    ----------
    slicer: :class:`compas_slicer.slicers.BaseSlicer`
        An instance of one of the compas_slicer.slicers classes.
    threshold: float
        Controls the degree of polyline simplification.
        Low threshold removes few points, high threshold removes many points.
    """

    logger.info("Paths simplification rdp")
    remaining_pts_num = 0

    with progressbar.ProgressBar(max_value=len(slicer.layers)) as bar:
        for i in range(slicer.number_of_layers):
            layer=slicer.get_layer(i)
            if not layer.is_raft:  # no simplification necessary for raft layer
                for ii,path in enumerate(layer.get_path()):
                    #print('layer',i,'path',ii,len(path.gradients),len(path.points))
                    # if i==0 or i ==1 or i==2:
                    #     print(path.gradients[:2],path.gradients[-2:],path.points[:2],path.points[-2:])
                    mask = rdp.rdp(np.array([point['point'] for point in path.points]), epsilon=threshold,return_mask=True)
                    #print(len(mask))
                    #print(path.points)
                    pts_rdp= [point for point, m in zip(path.points, mask) if m]
                    path.points = [{'point':Point(pt['point'][0], pt['point'][1], pt['point'][2]),'gradient':pt['gradient']} for pt in pts_rdp]
                    #print(path.gradients[0],len(path.gradients),len(path.points),len(mask))
                    remaining_pts_num += len(path.points)
                    bar.update(i)
        logger.info('%d Points remaining after rdp simplification' % remaining_pts_num)

def seams_smooth_with_gr(slicer, smooth_distance):
    """Smooths the seams (transition between layers)
    by removing points within a certain distance.

    Parameters
    ----------
    slicer: :class:`compas_slicer.slicers.BaseSlicer`
        An instance of one of the compas_slicer.slicers classes.
    smooth_distance: float
        Distance (in mm) to perform smoothing
    """

    logger.info("Smoothing seams with a distance of %i mm" % smooth_distance)

    for i, layer in enumerate(slicer.layers):
        if len(layer.paths) == 1 or isinstance(layer, compas_slicer.geometry.VerticalLayer):
            for j, path in enumerate(layer.paths):
                if path.is_closed:  # only for closed paths
                    pt0 = path.points[0]['point']
                    gr0 = path.points[0]['gradient']
                    # only points in the first half of a path should be evaluated
                    half_of_path = path.points[:int(len(path.points)/2)]
                    for point in half_of_path:
                        if distance_point_point(pt0, point['point']) < smooth_distance:
                            # remove points if within smooth_distance
                            path.points.pop(0)
                        else:
                            # create new point at a distance of the
                            # 'smooth_distance' from the first point,
                            # so that all seams are of equal length
                            vect = Vector.from_start_end(pt0, point['point'])
                            vect.unitize()
                            new_pt = pt0 + (vect * smooth_distance)
                            new_gr = gr0
                            path.points.insert(0, {'point':new_pt,'gradient':new_gr})

                            path.points.pop(-1)

                              # remove last point
                            break

        else:
            logger.warning("Smooth seams only works for layers consisting out of a single path, or for vertical layers."
                           "\nPaths were not changed, seam smoothing skipped for layer %i" % i)
            
def slice_mesh(layer:VerticalLayer_dart,edges_cutting_data:dict,faces_cutting_data:dict,mesh:Mesh,weight:float,edges_needed):            
    paths=  layer.get_path()      
    for path_id,path in enumerate(paths):
        
        for j,(edge,t) in enumerate(zip(path.get_edges(),path.get_t())):

            edge_point1_scalar_field=mesh.vertex_attribute(key=edge[0],name='scalar_field_org')
            edge_point2_scalar_field=mesh.vertex_attribute(key=edge[1],name='scalar_field_org')
            if edge_point2_scalar_field>edge_point1_scalar_field:
                edge_point_up=edge[1]
                edge_point_down=edge[0]
            else:
                edge_point_up=edge[0]
                edge_point_down=edge[1]
            
            org_faces=mesh.edge_faces(u=edge[0],v=edge[1])
            #print(org_faces)
            org_faces_vertices=[]
            for face in org_faces:
                vertices_face=copy.deepcopy(mesh.face_vertices(face))
                vertices_face.remove(edge[0])
                vertices_face.remove(edge[1])
                org_faces_vertices.append(vertices_face[0])
            
 
            w=trimesh_split_edge(mesh=mesh,u=edge[0],v=edge[1],t=t)

            mesh.vertex[w]['scalar_field_org']=weight
            new_faces=mesh.vertex_faces(w)
            for face,face_vertex in zip(org_faces,org_faces_vertices):
                faces_cutting_data[face]=[]
                for new_face in new_faces:
                    
                    if face_vertex in mesh.face_vertices(new_face):
                        
                        faces_cutting_data[face].append(new_face)
            #print((edge_point_down,edge_point_up),w,t)
            #edges_cutting_data[(edge_point_down,edge_point_up)]=(w,t)
            

            path.apply_w_key(w,j)
            path.add_edge_up_down(j,edge_point_up,edge_point_down)
    for path in layer.get_path():
        for point_id,edge in enumerate(path.get_edges()):
      
            faces_up=mesh.edge_faces(u=path.get_edge_up_pt(point_id),v=path.get_keys()[point_id])
            faces_down=mesh.edge_faces(u=path.get_edge_down_pt(point_id),v=path.get_keys()[point_id])
            path.add_faces(faces_up=faces_up,faces_down=faces_down)
def List_has_intersection(list_a, list_b) -> bool:
    """

    
    para:
        list_a (List[Tuple[int, int]]): 第一个列表。
        list_b (List[Tuple[int, int]]): 第二个列表。
        
    返回:
        bool: 如果两个列表有交集，则返回True；否则返回False。
    """
    # 将较小的列表转换为集合，以提高查找效率
   
    set_a = set(list_a)
    
    # 检查ListB中是否有任何元素在set_a中
    for element in list_b:
        if element in set_a:
            return True
if __name__ == "__main__":
    pass
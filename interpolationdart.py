from compas_slicer.pre_processing.interpolation_slicing_preprocessor import InterpolationSlicingPreprocessor
from compas_slicer.pre_processing.interpolation_slicing_preprocessor import get_union_method
from compas_slicer.pre_processing import CompoundTarget
from compas_slicer.pre_processing.gradient_evaluation import GradientEvaluation
import logging
import os
from compas.datastructures import Mesh
from compas_slicer.pre_processing.preprocessing_utils import region_split as rs, \
    topological_sorting as topo_sort
from compas_slicer.pre_processing import get_existing_cut_indices, get_vertices_that_belong_to_cuts, \
    replace_mesh_vertex_attribute
import compas_slicer.utilities as utils
from compas_slicer.parameters import get_param
from gradient_evaluation_dart import GradientEvaluation_Dart
from compound_target_dart import CompoundTargetDart
from assign_vertex_distance_dart import assign_final_distance_to_mesh_vertices,\
    assign_z_distance_to_mesh_vertices,assign_interpolation_distance_to_mesh_vertices,\
    get_final_distance,assign_up_distance_to_mesh_vertices,assign_multi_distance_to_mesh_vertices,\
    assign_org_gra_distance_to_mesh_vertices,assign_gradient_descent_distance_to_mesh_vertices
from distances import print_list_with_details,save_nested_list
import matplotlib.pyplot as plt
import numpy as np
from gradient_descent import gradient_descent
logger = logging.getLogger('logger')

__all__ = ['InterpolationSlicingPreprocessor',
           'point_creat_compound_target',]
class DartPreprocesssor(InterpolationSlicingPreprocessor):
    def __init__(self, mesh, parameters, DATA_PATH,offset=True):
        super().__init__(mesh, parameters, DATA_PATH)
        self.offset=offset

    
    def d_create_compound_targets(self):
        """ Creates the target_LOW and the target_HIGH and computes the geodesic distances. """

        # --- low target
        geodesics_method = get_param(self.parameters, key='target_LOW_geodesics_method',
                                     defaults_type='interpolation_slicing')
        method, params = 'min', []  # no other union methods currently supported for lower target

        self.target_LOW = CompoundTargetDart(self.mesh, 'boundary', 1, self.DATA_PATH,
                                         union_method=method,
                                         union_params=params,
                                         geodesics_method=geodesics_method,Lowornot=True,notflip=False)

        # --- high target
        geodesics_method = get_param(self.parameters, key='target_HIGH_geodesics_method',
                                     defaults_type='interpolation_slicing')
        method, params = get_union_method(self.parameters)
        
        logger.info("Creating target with union type : " + method + " and params : " + str(params))
        self.target_HIGH = CompoundTargetDart(self.mesh, 'boundary', 2, self.DATA_PATH,
                                          union_method=method,
                                          union_params=params,
                                          geodesics_method=geodesics_method,Lowornot=False,notflip=False)
        self.target_HIGH.set_other_target(self.target_LOW)
        self.target_LOW.set_other_target(self.target_HIGH)

        # # --- uneven boundaries of high target
        # self.target_HIGH.offset = get_param(self.parameters, key='uneven_upper_targets_offset',
        #                                     defaults_type='interpolation_slicing')

        # self.target_HIGH.compute_uneven_boundaries_weight_max(self.target_LOW)


        # # --- uneven boundaries of low target
        # self.target_LOW.offset = get_param(self.parameters, key='uneven_lower_targets_offset',
        #                                     defaults_type='interpolation_slicing')
        # self.target_LOW.compute_uneven_boundaries_weight_max(self.target_HIGH)

        
        # if self.offset:

        #     print("offset")
        #     #self.target_LOW.scale=0.5
        #     #self.target_HIGH.scale=1.2
        #     #self.target_LOW.scale=1.5
        #     self.target_HIGH.offset_distances(self.target_LOW)
        #     self.target_LOW.offset_distances(self.target_HIGH)
        #     self.target_HIGH.offset_distances(self.target_LOW)
        #     self.target_LOW.offset_distances(self.target_HIGH,True)
        #     self.target_HIGH.offset_distances(self.target_LOW,True)
        #     self.target_LOW.save_offset_distance("distances_LOW_offset.json")
        #     self.target_HIGH.save_offset_distance("distances_HIGH_offset.json")        
          
        #  --- save intermediary get_distance outputs
        self.target_LOW.save_distances("distances_LOW.json")
        self.target_HIGH.save_distances("distances_HIGH.json")

    
    

    
    def create_gradient_evaluation(self, target_1=None, target_2=None, save_output=True,
                                   norm_filename='gradient_norm.json', g_filename='gradient.json',
                                   way='org',target_index=0,frame='',g_n=True,show_graph=False,scalar_field=None,guide_field=None):
        """
        target 1: lower boundary
        target 2: upper boundary
        g_n: if gradient evaluation will be down
        way: if scalar field will be done on z or on up or combining up and down
        Creates a compas_slicer.pre_processing.GradientEvaluation that is stored in self.g_evaluation
        Also, computes the gradient and gradient_norm and saves them to Json .
        """
        #assert self.target_LOW.VN == target_1.VN, "Attention! Preprocessor does not match targets. "
        if scalar_field is not None:
            for vkey in self.mesh.vertices():
                self.mesh.vertex[vkey]['scalar_field']=scalar_field[vkey]
        elif way == 'org':
            assign_interpolation_distance_to_mesh_vertices(self.mesh, weight=0.5,
                                                       target_LOW=target_1, target_HIGH=target_2)
        elif way == 'z':
            assign_z_distance_to_mesh_vertices(self.mesh, weight=0.5,
                                                       target_LOW=target_1, target_HIGH=target_2)
        elif way=='up':
            assign_up_distance_to_mesh_vertices(self.mesh, 
                                                        target_HIGH=target_2,target_index=target_index)
        elif way=='down':
            assign_up_distance_to_mesh_vertices(self.mesh, 
                                                        target_HIGH=target_1,target_index=target_index)
        elif way=='multi':
            inter=frame/50
            assign_multi_distance_to_mesh_vertices(self.mesh, weight=0.5,
                                                       target_LOW=target_1, target_HIGH=target_2,inter=inter)
        elif way=='org_gra':
            assign_org_gra_distance_to_mesh_vertices(self.mesh,
                                                       target_LOW=target_1, target_HIGH=target_2)
        elif way=='gradient_descent':
    
            assign_gradient_descent_distance_to_mesh_vertices(self.mesh,
                                                       target_LOW=target_1, target_HIGH=target_2,guide_field=guide_field)
        
    
        else:
            assign_final_distance_to_mesh_vertices(self.mesh, weight=0.5,
                                                       target_LOW=target_1, target_HIGH=target_2)
            

        if g_n:
        
            g_evaluation = GradientEvaluation_Dart(self.mesh, self.DATA_PATH)
            g_evaluation.compute_gradient()
            g_evaluation.compute_gradient_norm() 

            if save_output:
                # save results to json
                utils.save_to_json(g_evaluation.vertex_gradient_norm, self.OUTPUT_PATH, norm_filename)
                utils.save_to_json(utils.point_list_to_dict(g_evaluation.vertex_gradient), self.OUTPUT_PATH, g_filename)
  
                scalar_field=[]
                for vertex,data in self.mesh.vertices(data=True):
                    scalar_field.append(data['scalar_field'])
                print_list_with_details(scalar_field,"scalar field")
                save_nested_list(file_path=self.OUTPUT_PATH,file_name="scalar_field"+".josn",nested_list=scalar_field)
            if show_graph:
                self.target_HIGH
            
                distance1=self.target_HIGH.get_avg_distances_from_other_target(self.target_LOW)
                distance2=self.target_LOW.get_avg_distances_from_other_target(self.target_HIGH)
                g_avg=np.average(g_evaluation.vertex_gradient_norm)
                data=[g_avg/(g) for g in g_evaluation.vertex_gradient_norm]
                # 创建直方图
                bin_size = 0.2
                min_range =0
                max_range = 2
                bins = np.arange(min_range, max_range + bin_size, bin_size)
                n, bins, patches = plt.hist(data, bins=bins, alpha=0.75)
                
                # 设置y轴的最大值
                max_y = 30000  # 你可以根据需要调整这个值
                plt.ylim(0, max_y)
                # 添加标题和轴标签
                plt.title('数据分布直方图')
                plt.xlabel('值')
                plt.ylabel('频率')

                # 显示网格
                plt.grid(True)
                
                # 在每个直方上标注具体数据
                for i in range(len(patches)):
                    x = (bins[i] + bins[i+1]) / 2
                    y = n[i]
                    if y > max_y:
                        y = max_y - 0.5  # 如果高度超过了最大值，将标签位置稍微下调
                    plt.text(x, y, str(int(y)), ha='center', va='bottom')

                # 显示图表
                plt.show()                
                

            return g_evaluation  
    def create_all_distance_fields_evalueation(self, target_1:CompoundTargetDart, target_2:CompoundTargetDart, save_output=True,
                                   norm_filename='all_field_gradient_norm', g_filename='all_field_gradient',
                                    target_index=0,frame='',g_n=True,show_graph=False):
        g_evaluations={'low':[],'high':[]}
        for boundary_i,field in enumerate(self.target_LOW._distances_lists):
            #print(boundary_i,field)
            for vkey in self.mesh.vertices():
                self.mesh.vertex[vkey]['scalar_field']=field[vkey]
                #print(boundary_i,vkey,self.mesh.vertex[vkey]['scalar_field'])
            g_evaluation = GradientEvaluation_Dart(self.mesh, self.DATA_PATH)
            g_evaluation.compute_gradient()
            g_evaluation.compute_gradient_norm()
            g_evaluations['low'].append(g_evaluation)
            utils.save_to_json(g_evaluation.vertex_gradient_norm, self.OUTPUT_PATH, 
                               norm_filename+'_low'+str(boundary_i)+'.json')
            utils.save_to_json(utils.point_list_to_dict(g_evaluation.vertex_gradient), 
                               self.OUTPUT_PATH, g_filename+'_low'+str(boundary_i)+'.json')
        for boundary_i,field in enumerate(self.target_HIGH._distances_lists):
            for vkey in self.mesh.vertices():
                self.mesh.vertex[vkey]['scalar_field']=field[vkey]
                #print(boundary_i,vkey,self.mesh.vertex[vkey]['scalar_field'])
            g_evaluation = GradientEvaluation_Dart(self.mesh, self.DATA_PATH)
            g_evaluation.compute_gradient()
            g_evaluation.compute_gradient_norm()
            g_evaluations['high'].append(g_evaluation)
            utils.save_to_json(g_evaluation.vertex_gradient_norm, self.OUTPUT_PATH, 
                               norm_filename+'_high'+str(boundary_i)+'.json')
            utils.save_to_json(utils.point_list_to_dict(g_evaluation.vertex_gradient), 
                               self.OUTPUT_PATH, g_filename+'_high'+str(boundary_i)+'.json')

        return g_evaluations

    def find_critical_points(self, g_evaluation:GradientEvaluation_Dart, output_filename,sort_by_height=False):
        """ Computes and saves to json the critical points of the df on the mesh (minima, maxima, saddles)"""
        g_evaluation.find_critical_points()
        # save results to json
        if sort_by_height:
            heights=[]
            for saddle in g_evaluation.saddles:
                height=self.mesh.vertex_attribute(saddle, 'z')
                heights.append(height)
            saddles_sorted=sorted(zip(heights,g_evaluation.saddles),reverse=True)
            saddles_sorted=[saddle for _,saddle in saddles_sorted]
            g_evaluation.saddles=saddles_sorted


        
        utils.save_to_json(g_evaluation.saddles, self.OUTPUT_PATH, output_filename)

        return(g_evaluation.saddles) 
    
    def find_critical_points_with_related_boundary(self, g_evaluation:GradientEvaluation_Dart, output_filename,sort_by_height=False):
        """ Computes and saves to json the critical points of the df on the mesh (minima, maxima, saddles)"""
        g_evaluation.find_critical_points_with_related_boundary()
        # save results to json
        if sort_by_height:
            heights=[]
            for saddle in g_evaluation.saddles:
                height=self.mesh.vertex_attribute(saddle, 'z')
                heights.append(height)
            saddles_sorted=sorted(zip(heights,g_evaluation.saddles),reverse=True)
            saddles_sorted=[saddle for _,saddle in saddles_sorted]
            g_evaluation.saddles=saddles_sorted


        
        
        for i in g_evaluation.saddles:
            max_vs=g_evaluation.related_boundary[i]['max_v']
            min_vs=g_evaluation.related_boundary[i]['min_v']
            g_evaluation.related_boundary[i]['target_high']=set()
            g_evaluation.related_boundary[i]['target_low']=set()
            for max_v in max_vs:
                related_v = gradient_descent(self.mesh,max_v,'scalar_field',True)
                for bi,boundary in enumerate(self.target_HIGH.clustered_vkeys):
                    for vi in boundary:
                        if vi==related_v:
                            g_evaluation.related_boundary[i]['target_high'].add(bi)
                            break
            for min_v in min_vs:
                related_v = gradient_descent(self.mesh,min_v,'scalar_field',False)
                for bi,boundary in enumerate(self.target_LOW.clustered_vkeys):
                    for vi in boundary:
                        if vi==related_v:
                            g_evaluation.related_boundary[i]['target_low'].add(bi)
                            break
            
            print('saddle point related boundary:',i,g_evaluation.related_boundary[i]['target_high'],g_evaluation.related_boundary[i]['target_low'])
        saddles_org=g_evaluation.saddles.copy()
        for i in saddles_org:
            if len(g_evaluation.related_boundary[i]['target_high'])==1 and len(g_evaluation.related_boundary[i]['target_low'])==1:
                print('remove_saddle_point',i)
                g_evaluation.saddles.remove(i)
            else:
                print(i,)
        utils.save_to_json(g_evaluation.saddles, self.OUTPUT_PATH, output_filename)
        return(g_evaluation.saddles) 
    
    def load_scalar_field(self):
        from distances import load_nested_list
        scalar_field=load_nested_list(self.OUTPUT_PATH,'scalar_field.json')
        for vkey in self.mesh.vertices():
            self.mesh.vertex[vkey]['scalar_field']=scalar_field[vkey]
    def save_scalar_field(self):
        scalar_field=[]
        for vkey in self.mesh.vertices():
            scalar_field.append(self.mesh.vertex[vkey]['scalar_field'])
        utils.save_to_json(filepath=self.OUTPUT_PATH,name="scalar_field.json",data=scalar_field)
    def save_offset_distance(self):
        self.target_LOW.save_offset_distance("distances_LOW_offset.json")
        self.target_HIGH.save_offset_distance("distances_HIGH_offset.json")

    def load_offset_distance(self):
        self.target_HIGH.load_offset_distance("distances_HIGH_offset.json")
        self.target_LOW.load_offset_distance("distances_LOW_offset.json")


    



    # def creat_saddle_evaluation(self, target_1, target_2=None, save_output=True,
    #                                norm_filename='gradient_norm.json', g_filename='gradient.json',way='org'):
    #     self.target_HIGH.point_offset_distance_High()
    #     pass
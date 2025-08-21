from compas.datastructures import Mesh,trimesh_split_edge
from compas_slicer.pre_processing import CompoundTarget
import scipy.stats
from interpolationdart import DartPreprocesssor
import os
from compas.datastructures import Mesh
import logging
import compas_slicer.utilities as utils
import time
from interpolationdart import DartPreprocesssor
import progressbar
import numpy as np
import igl
from compound_target_dart import save_nested_list,load_nested_list,mesh_smooth_centroid_vertices_influence
from interpolation_slicer_dart import InterpolationSlicer_Dart,simplify_paths_rdp_with_gr,seams_smooth_with_gr
from compas_slicer.pre_processing.preprocessing_utils.geodesics import get_igl_EXACT_geodesic_distances
from compound_target_dart import CompoundTargetDart
import scipy
import math
import copy
import time
from typing import List,Tuple,Optional,Set,Dict
from distances import multi_source_dijkstra
import heapq
from layer_dart import path_dart
from distances import save_nested_list,load_nested_list
from compas.files import OBJWriter
from layer_dart import VerticalLayer_dart
from gradient_evaluation_dart import GradientEvaluation_Dart
import networkx as nx
from collections import deque, defaultdict
def main():
    avg_layer_height=15
    input_folder_name='beam1B'#'table_2''example_jun_bg''whole''data_Y_shape' 'data_vase''data_costa_surface''data_Y_shape_o''data_vase_o''data_costa_surface_o''Jun_ab_testmultipipe'
    #'Jun_ah_testb''Jul_ai''Jul_h''Jul_I''Jul_ab''Jul_ah''Jul_ba''table_1''Aug_ac_ex''Aug_bg''Aug_bh''example_jun_bg'
    DATA_PATH = os.path.join(os.path.dirname(__file__), input_folder_name)
    OUTPUT_PATH = utils.get_output_directory(DATA_PATH)
    OBJ_INPUT_NAME = os.path.join(DATA_PATH, 'mesh.obj')
    from compas_slicer.pre_processing import create_mesh_boundary_attributes
    mesh = Mesh.from_obj(os.path.join(OUTPUT_PATH,"edited_mesh.obj"))  
    print("old mesh")
    # --- Load targets (boundaries)
    low_boundary_vs = utils.load_from_json(DATA_PATH, 'boundaryLOW.json')
    high_boundary_vs = utils.load_from_json(DATA_PATH, 'boundaryHIGH.json')
    create_mesh_boundary_attributes(mesh, low_boundary_vs, high_boundary_vs)    
    parameters = {
        'avg_layer_height': avg_layer_height,  # controls number of curves that will be generated
    }
    preprocessor = DartPreprocesssor(mesh, parameters, DATA_PATH)
    preprocessor.d_create_compound_targets() 
    preprocessor.load_scalar_field()
    preprocessor.load_offset_distance()
    saddles=load_nested_list(file_path=OUTPUT_PATH, file_name= 'saddles.json')

    cutter=mesh_cutter(mesh=mesh,target01=preprocessor.target_LOW,target02=preprocessor.target_HIGH,saddles=saddles,
                       processor=preprocessor,Outputpath=OUTPUT_PATH,parameters=parameters,use_memory=True)
    cutter.cut_mesh()
    cutter.slice_segment()
class mesh_cutter:
    def __init__(self,mesh:Mesh,saddles,target01:CompoundTargetDart,target02:CompoundTargetDart,processor:DartPreprocesssor,Outputpath,parameters,G:Optional[GradientEvaluation_Dart]=None,animation_frame='',use_memory=False) -> None:
        self.mesh=mesh
        self.saddles=saddles
        self.target_low=target01
        self.target_high=target02
        self.preprocessor = processor
        self.target_high
        self.Outputpath=Outputpath
        self.VN = len(list(self.mesh.vertices()))
        self.number_of_saddles=len(list(self.saddles))
        self.parameters=parameters
        self.G=G
        self.distances_from_saddles=[]
        self.distances_from_saddles_flipped=[]
   

        self.saddles_facing_High=[]
        self.saddles_facing_Low=[]
        self.saddles_facing_High_vertices=[]
        self.saddles_facing_Low_vertices=[]

        self.get_saddles_direction()
        self.get_average_slice_field_difference()


        
        self.number_of_saddles_targets=self.number_of_saddles+self.target_high.number_of_boundaries+self.target_low.number_of_boundaries
        self.influence_manual_list=[0]*self.number_of_saddles_targets
        self.high_list=[0]*self.number_of_saddles_targets
        self.items_target_high=list(range(self.number_of_saddles,self.number_of_saddles+self.target_high.number_of_boundaries))
        self.items_target_low=list(range(self.number_of_saddles+self.target_high.number_of_boundaries,self.number_of_saddles_targets))
        self.items_saddles=list(range(self.number_of_saddles))
        self.animation_frame=animation_frame

        self.edges_cutting_data={}
        self.faces_cutting_data={}
        self.new_mesh=self.mesh.copy()
        self.use_memory=use_memory
    def get_average_slice_field_difference(self):
        if max(self.target_high.offset_distance_list)>0:
            print("get number of layers from offset distance")
            self.average_slice_field_difference=0
            max_distance=0
            sum_distance=0
            N_bv=0
            for boundary in self.target_high.clustered_vkeys:
                #print(boundary,"boundary")
                for vkey in boundary:
                    max_distance=max(max_distance,self.target_low.offset_distance_list[vkey])
                    N_bv+=1
                    sum_distance+=self.target_low.offset_distance_list[vkey]
            for boundary in self.target_low.clustered_vkeys:
                #print(boundary,"boundary")
                for vkey in boundary:
                    max_distance=max(max_distance,self.target_high.offset_distance_list[vkey])
                    N_bv+=1
                    sum_distance+=self.target_high.offset_distance_list[vkey]
            avg_distance=sum_distance/N_bv

            print("max distance",max_distance,'avg_distance',avg_distance)
            number_of_layers=int(avg_distance/self.parameters['avg_layer_height']+0.5)
            self.average_slice_field_difference=1/number_of_layers
            # self.max_distance=max_distance
            # self.layer_height=max_distance/number_of_layers
            self.max_distance=avg_distance
            self.layer_height=avg_distance/number_of_layers
        elif self.G is not None:
            print('get number of layers from gradient norm')
            self.max_distance=1/np.mean(self.G.vertex_gradient_norm)
            number_of_layers=int(self.max_distance/self.parameters['avg_layer_height']+0.5)
            self.average_slice_field_difference=1/number_of_layers
            self.layer_height=self.max_distance/number_of_layers
        else:
            raise Exception('can not get number of layers')    
            
    
      
   
    def get_saddles_direction(self):
        """
        将鞍点分为朝上与朝下的
        mark the saddles points with up facing and low facing
        """
        for i,saddle in enumerate(self.saddles):
         
            normal=self.mesh.vertex_normal(key=saddle)[2]
            if normal>0:
                self.saddles_facing_High.append(i)
              
                self.saddles_facing_High_vertices.append(saddle)
            else:
                self.saddles_facing_Low.append(i)
                self.saddles_facing_Low_vertices.append(saddle)
    
    def cut_mesh(self):
        
        #self.find_saddles_adjacent_path()
        if self.use_memory:
            # 1/0
            # this part has a bug because the paths are sorted
            self.cutted_mesh = Mesh.from_obj(os.path.join(self.Outputpath,"cutted_mesh.obj"))  
            segment_str_keys=load_nested_list(file_path=self.Outputpath,file_name='segment.json')
    
            self.segment= {eval(k) if isinstance(k, str) and ',' in k else k: v for k, v in segment_str_keys.items()}
            self.slicer=InterpolationSlicer_Dart.load_slicer_from_file(self.Outputpath,'cutter_slicer.json')
            self.saddles_and_weights=utils.load_from_json(self.Outputpath,'saddles_and_weights.json')
        else:
            print('new segmentation')
            self.creat_cutting_lines()
            self.get_saddle_paths_geodesic_distance()
            segment_str_keys = {str(k): v for k, v in self.segment.items()}
            save_nested_list(file_path=self.Outputpath,file_name='segment.json',nested_list=segment_str_keys)
            self.slicer.save_slicer(self.Outputpath,'cutter_slicer.json')
            utils.save_to_json(self.saddles_and_weights,self.Outputpath,'saddles_and_weights.json')
            obj_writer = OBJWriter(filepath= os.path.join(self.Outputpath, "cutted_mesh.obj"), meshes=[self.new_mesh])
            obj_writer.write()
    

        
    def creat_cutting_lines(self):
        saddles_weithts_up=[{'direction':True,'weight':self.mesh.vertex_attribute(name='scalar_field',key=saddle_vkey)+0.00001,'vertex':saddle_vkey,'saddle':saddle_ind} for saddle_ind,saddle_vkey in zip(self.saddles_facing_High,self.saddles_facing_High_vertices)]
        saddles_weights_down=[{'direction':False,'weight':self.mesh.vertex_attribute(name='scalar_field',key=saddle_vkey)-0.00001,'vertex':saddle_vkey,'saddle':saddle_ind} for saddle_ind,saddle_vkey in zip(self.saddles_facing_Low,self.saddles_facing_Low_vertices)]
        saddles_and_weights=saddles_weithts_up+saddles_weights_down

        slicer = InterpolationSlicer_Dart(self.new_mesh, self.preprocessor, self.parameters)
        
        
        slicer.get_scalar_field_without_weigh()
        saddles_and_weights.sort(key=lambda x:x['weight'])
        saddle_weights= [x['weight'] for x in saddles_and_weights]
        slicer.generate_paths_with_weights(saddle_weights,output_edge=True,
                                           saddles=[x['vertex'] for x in saddles_and_weights],
                                           edge_edit_data=self.edges_cutting_data,
                                           face_edit_data=self.faces_cutting_data,)
        self.slicer=slicer
        self.saddles_and_weights=saddles_and_weights
        self.VN_old=self.VN
        self.cutted_mesh=self.new_mesh
        self.VN=len(list(self.new_mesh.vertices()))   
        # obj_writer = OBJWriter(filepath= os.path.join(self.Outputpath, "cutted_mesh.obj"), meshes=[self.new_mesh])
        # obj_writer.write()  
        utils.save_to_json(slicer.to_data(), self.Outputpath, 'curved_slicer_saddle.json')   
      
        return slicer,saddles_and_weights

    # def find_saddles_adjacent_path(self):
    #     slicer=self.slicer
    #     saddles_and_weights=self.saddles_and_weights
    #     for i,saddle_and_weight in enumerate(saddles_and_weights):
    #         saddle_vkey=saddle_and_weight['vertex']
    #         saddle_neibour_edges=self.mesh.vertex_edges(saddle_vkey)
    #         layer=slicer.get_layer(i)
    #         saddle_paths=layer.get_path()
    #         for saddle_path in saddle_paths:
    #             path_edges=saddle_path.get_edges()
    #             if not List_has_intersection(saddle_neibour_edges,path_edges):
    #                 layer.remove_path(saddle_path)   
    #     return slicer

    def get_saddle_paths_geodesic_distance(self):
        slicer=self.slicer
        saddles_and_weights=self.saddles_and_weights
        self.segement_relationship=set()
        # new_mesh=copy.deepcopy(self.mesh)
        failure_edge=[]


        # for key in self.faces_cutting_data:
        #     items=self.faces_cutting_data[key]
        #     print(key,items)
        #     if int(key) in items:
        #         print('face multiple',key,items)
        expand_dict_values(self.faces_cutting_data)
        #print(faces_cutting_data)
        for layer_id in range(slicer.number_of_layers):
            layer=slicer.get_layer(layer_index=layer_id)
            for path in layer.get_path():
                path.update_faces(self.faces_cutting_data)
                    
              
        save_nested_list(self.Outputpath,"fail_edge.json",failure_edge)

                
        print(len(list(self.mesh.vertices())),len(list(self.new_mesh.vertices())))
        #mesh_topo=[{} for _ in range(self.VN)]
        segment={}
        
        for layer_id in range(slicer.number_of_layers):
            print(layer_id,'get_layer_field')
            layer=slicer.get_layer(layer_index=layer_id)
            mesh_delet_up=copy.deepcopy(self.new_mesh)
            mesh_delet_down=copy.deepcopy(self.new_mesh)
            self.delet_layer_faces(mesh_delet_up,layer_id,True)
            self.delet_layer_faces(mesh_delet_down,layer_id,False) 
            # obj_writer = OBJWriter(filepath= os.path.join(self.Outputpath, str(layer_id)+"cutted_mesh_up.obj"), meshes=[mesh_delet_up])
            # obj_writer.write()
            # obj_writer = OBJWriter(filepath= os.path.join(self.Outputpath, str(layer_id)+"cutted_mesh_down.obj"), meshes=[mesh_delet_down])            
            # obj_writer.write()
              
            paths=layer.get_path()
            up_or_down=saddles_and_weights[layer_id]['direction']
            if up_or_down:
                #branch is facing up so for upper we need to caculate seperat
                for path_id,path in enumerate(paths):
                    path_distance_field,connection,layer_path=self.igl_topo(mesh=mesh_delet_down,source=[path])
                    add_to_seg(segment,path_distance_field,connection,layer_path,'lower_dist')
                path_distance_field,connection,layer_path=self.igl_topo(mesh=mesh_delet_up,source=paths)
                add_to_seg(segment,path_distance_field,connection,layer_path,'upper_dist')
            else:
                for path_id,path in enumerate(paths):
                    path_distance_field,connection,layer_path=self.igl_topo(mesh=mesh_delet_up,source=[path])
                    add_to_seg(segment,path_distance_field,connection,layer_path,'upper_dist')
                path_distance_field,connection,layer_path=self.igl_topo(mesh=mesh_delet_down,source=paths)
                add_to_seg(segment,path_distance_field,connection,layer_path,'lower_dist')
        
        bad_layer_ids=[]
        print(segment.keys())
        for layer_path in segment.keys():
            if len (layer_path)==1:
                layer_path=layer_path[0]
                bad_layer_ids.append(layer_path[0])
                print('bad_layer_path',layer_path,layer_path[0])
        if len(bad_layer_ids)>0:
            print('bad_layer_ids',bad_layer_ids)
            self.slicer.remove_layers(bad_layer_ids)
            bad_layer_ids.sort(reverse=True)
            for layer_id in bad_layer_ids:
                saddles_and_weights.pop(layer_id)
            self.get_saddle_paths_geodesic_distance()
        else:
            self.segment=segment

    
    def delet_layer_faces(self,Mesh:Mesh,layer_id:int,direction:True):
        #print("mesh_cutting 207",direction)
        """
        mesh:mesh with cutting line
        layer_id: saddle path's layer's index
        directino: True delet up faces and lower layer's lower faces
                    False delet low faces and upper layer's upper faces
        """
        for layer_j in range(self.slicer.number_of_layers):
            error1=[]
            error2=[]
            error3=[]
            if layer_j==layer_id:
                layer=self.slicer.get_layer(layer_j)
                for path in layer.get_path():
                    if direction:
                        faces=path.faces_up
                    else:
                        faces=path.faces_down
                    for face in faces:
                       
                        try:
                            Mesh.delete_face(face)  
                        except:
                            error1.append(face)
                          
            elif layer_j>layer_id and not direction:
                layer=self.slicer.get_layer(layer_j)
                for path in layer.get_path():
                    faces=path.faces_up
                    for face in faces:
                        try:
                            Mesh.delete_face(face)
                        except:
                            error2.append(face)    
            elif layer_j<layer_id and direction:
                layer=self.slicer.get_layer(layer_j)
                for path in layer.get_path():
                    faces=path.faces_down
                    for face in faces:
                     
                        try:
                            Mesh.delete_face(face)
                        except:
                            error3.append(face)
        #print(len(error1),len(error2),len(error3))
    
    def get_selected_faces(self):
        pass

    def get_new_mesh(self,face_items,vertices):
        return self.mesh.from_vertices_and_faces(faces=face_items,vertices=vertices)
    def igl_topo(self,mesh:Mesh,source:List[path_dart],way='igl'):

        v_start=[v for path in source for v in path.get_keys()]
        if way=='igl':
            distances=get_igl_EXACT_geodesic_distances(mesh=mesh,vertices_start=v_start)
            distances=distances.tolist()  
            topo_conection = [d>0 for d in distances]
            for i,conect in enumerate(topo_conection):
                if i in v_start:
                    topo_conection[i]=True
                    distances[i]=0            
        else:
            distances=multi_source_dijkstra(mesh=mesh,sources=v_start)
            
            topo_conection = [not math.isinf(d) for d in distances]
        layer_path=self.check_topo(topo_conection)

        #finite_count = sum(1 for d in topo_conection if d) 
        #print("igl_topo",finite_count,len(topo_conection)-finite_count,max(distances))
        #print("igl_topo",len(distances),len(topo_conection))
        return distances,topo_conection,layer_path      

    def check_topo(self,topo_conection):
        conected_vkeys=[]
        for vkey in range(self.VN):
            if topo_conection[vkey]==True:
                conected_vkeys.append(vkey)
        #print(conected_vkeys)
        conected_paths=[]
     
            
        for layer_id in range(self.slicer.get_layer_number):
            layer=self.slicer.get_layer(layer_index=layer_id)
            for path_id,path in enumerate(layer.get_path()):
                path_keys=path.get_keys()
                #print(path_keys)
                if List_has_intersection(path_keys,conected_vkeys):
                    conected_paths.append((layer_id,path_id)) 
        for b_id,boundary in enumerate(self.target_high.clustered_vkeys):
            #print("checkb",len(boundary))
            if List_has_intersection(boundary,conected_vkeys):
                conected_paths.append((-1,b_id))
        for b_id,boundary in enumerate(self.target_low.clustered_vkeys):
            #print("checkb",len(boundary))
            if List_has_intersection(boundary,conected_vkeys):
                conected_paths.append((-2,b_id))
        #print(conected_paths)
        for layer_path in conected_paths:
            if layer_path[0]>=0:
                layer=self.slicer.get_layer(layer_path[0])
                paths=layer.get_path()
                path=paths[layer_path[1]]
                path=path.get_keys()
            elif layer_path[0]==-1:
                path=self.target_high.clustered_vkeys[layer_path[1]]
            elif layer_path[0]==-2:
                path=self.target_low.clustered_vkeys[layer_path[1]]
            else:

                print("illigal layer name",layer_path)
            if not List_in_List(path,conected_vkeys):
                print('bug',layer_path)
        return tuple(conected_paths) 
    
    def slice_segment(self):
        #print(self.segment.keys())
        all_layer_height=[]
        self.topo_of_thin_segements()
        slicer_name=[]
        for segment_id,key in enumerate(self.segment.keys()):
            self.VN
            mask=self.segment[key]['mask']
            new_mesh=get_segment_mesh(self.cutted_mesh,mask)
            VN=len(list(new_mesh.vertices()))
            print('start slicing seg:',segment_id,'layer_path',key,VN)
            boundary_seg=0
            if key[-1][0]==-1:
                boundary_seg=1
                dist_low=self.segment[key]['lower_dist']
                dist_high=self.target_high._distances_lists[key[-1][1]]
                #dist_high=get_igl_EXACT_geodesic_distances(mesh=new_mesh,vertices_start=self.target_high.clustered_vkeys[key[-1][1]])
                #print('high',len(dist_high))
                for d_id,d in enumerate(dist_high):
                    if not mask[d_id]:
                        dist_high[d_id]=0                
            elif key[-1][0]==-2:
                boundary_seg=2
                dist_high=self.segment[key]['upper_dist']
                dist_low=self.target_low._distances_lists[key[-1][1]]
                # if len(dist_low)!=self.VN_old:
                #     print('maybe bug')
                #print('low')           
                #dist_low=get_igl_EXACT_geodesic_distances(mesh=new_mesh,vertices_start=self.target_low.clustered_vkeys[key[-1][1]])
                #print('low',len(dist_low))  
                for d_id,d in enumerate(dist_low):
                    if not mask[d_id]:
                        dist_low[d_id]=0
                
            else:
                print(self.segment[key].keys())
                dist_low=self.segment[key]['lower_dist']
                dist_high=self.segment[key]['upper_dist']
            boundary_dists=[]
            for m,l,h in zip(mask,dist_low,dist_high):
                if m:
                    if l==0:
                        boundary_dists.append(h)
                    if h==0:
                        boundary_dists.append(l)
            max_dist=np.mean(boundary_dists)
    

            g_e=assign_scalar_field(new_mesh,dist_high,dist_low,mask,self.Outputpath,boundary_seg)
            
          
            utils.save_to_json(g_e.vertex_gradient_norm, self.Outputpath,'gradient_norm'+str(segment_id)+'.json')
            utils.save_to_json(utils.point_list_to_dict(g_e.vertex_gradient), self.Outputpath, 'gradient'+str(segment_id)+'.json')
            obj_writer = OBJWriter(filepath= os.path.join(self.Outputpath, "cutted_mesh"+str(segment_id)+".obj"), meshes=[new_mesh])
            obj_writer.write()
            self.save_dist_list(dist_low,mask,'dist_low'+str(segment_id)+'.json')
            self.save_dist_list(dist_high,mask,'dist_high'+str(segment_id)+'.json')
            self.save_scalar_field(new_mesh,'scalar_field'+str(segment_id)+'.json')
            slicer=InterpolationSlicer_Dart(mesh=new_mesh,slice_on_boundary=False,gradient_evaluation=g_e)
            # max_dist=1/np.median(g_e.vertex_gradient_norm)
            
            
            Estimated_number_of_layers=max_dist/self.layer_height
            layer_height,params_list=self.compute_params_list(self.segment[key],max_dist)
            layer_heights_estimated=[1/(norm*Estimated_number_of_layers) for norm in g_e.vertex_gradient_norm]
            utils.save_to_json(layer_heights_estimated, self.Outputpath,'layer_height_vertices'+str(segment_id)+'.json')
            
            print(max_dist,max(dist_high),max(dist_low),'_________segment distances__________',Estimated_number_of_layers,layer_height)
            #n=int(max_dist/self.parameters['avg_layer_height'])+1
            print(segment_id,params_list)
            #get_number_of_layers(params_list)
            if params_list is not None and params_list!=[]:
                all_layer_height.extend(layer_heights_estimated)
                slicer.generate_paths(params_list=params_list)
                # simplify_paths_rdp_with_gr(slicer, threshold=0.25)
                # seams_smooth_with_gr(slicer, smooth_distance=3)
                utils.save_to_json(slicer.to_data(), self.Outputpath, 'segment_slicer'+str(segment_id)+str(self.animation_frame)+'.json')
                slicer_name.append('segment_slicer'+str(segment_id)+str(self.animation_frame)+'.json')
                print('save','segment_slicer'+str(segment_id)+str(self.animation_frame)+'.json',' to ',self.Outputpath)
            else:
                print('No layers have been saved in the json file for segment',segment_id)
        utils.save_to_json(slicer_name, self.Outputpath, 'slicer_names.json')
        utils.save_to_json(all_layer_height, self.Outputpath,'all_layer_height.json')
        avg_layer_height=np.mean(all_layer_height)
        utils.save_to_json([(height-avg_layer_height)/avg_layer_height for height in all_layer_height], self.Outputpath,'layer_height_error.json')

    def save_dist_list(self,list:List[float],mask:List[bool],name:str):
        new_list=[]
        for i,ifin in zip(list,mask):
            if ifin:
                new_list.append(i)
        save_nested_list(file_path=self.Outputpath,file_name=name,nested_list=new_list)
    def save_scalar_field(self,mesh:Mesh,name:str):
        scalar_field=[]
        for vertex in mesh.vertices():
            scalar_field.append(mesh.vertex[vertex]['scalar_field'])
        save_nested_list(file_path=self.Outputpath,file_name=name,nested_list=scalar_field)
    def topo_of_thin_segements(self):
        self.get_seg_width()
        groups=self.find_thin_segment_groups()
        layer_values={}
        for group in groups:
            print('conected_group',group.keys())
            root=self.get_group_root(group)
            print('root',root)
            inter_values_=self.apply_layer_path_inter_value(group,root)
            print('inter_values',inter_values_)
            
            layer_values.update(inter_values_)
        print('inter_values_whole',layer_values)
        self.assign_segmentation_inter_values(layer_values)
   
    def get_seg_width(self):
        for key in self.segment.keys():
            layer1=-3
            layer2=-3
            for layer,path in key:
                if layer1==-3:
                    layer1=layer
                    path1=path
                elif layer2==-3 and layer!=layer1:
                    layer2=layer
                    path2=path
                elif layer not in (layer1,layer2):
                    print("bug: more than two layers in one segment")
            self.segment[key]['width']=(self.get_layer_path_scalar_field(layer2,path2)-self.get_layer_path_scalar_field(layer1,path1))
            if self.segment[key]['width']<0:
                self.segment[key]['width']=-self.segment[key]['width']
                self.segment[key]['layers']=(layer2,layer1)
            else:
                self.segment[key]['layers']=(layer1,layer2)
        print(self.segment.keys())
        self.thin_segment={}
        for key in self.segment.keys():
            if self.segment[key]['width']<5*self.average_slice_field_difference:
                self.thin_segment[key]=self.segment[key]
                print(self.thin_segment[key]['layers'])
        print(self.thin_segment.keys())
    def compute_params_list(self,segment,seg_max_dist):
        inter_value1=segment['inter_values'][0]
        inter_value2=segment['inter_values'][1]

        inter_value1=inter_value1%1

        inter_value2=inter_value2%1
        print(inter_value1,1-inter_value2,seg_max_dist,self.layer_height)
        return(split_interval(0,seg_max_dist,inter_value1,1-inter_value2,self.layer_height))
        width=segment['width']
        if width<5*self.average_slice_field_difference:  
            return(calculate_ratios(self.average_slice_field_difference,inter_value1,inter_value2+segment['width']))
        else:
            offset_dist1=(inter_value1*self.max_distance)%self.layer_height
            offset_dist2=(inter_value2*self.max_distance)%self.layer_height
            print(segment['layers'],"offset_dist1",offset_dist1,'offset_dist2',offset_dist2)
            Pa_Pb=seg_max_dist+offset_dist1-offset_dist2
            number_of_segments=math.ceil(Pa_Pb/self.layer_height+0.5)
            start_value=(inter_value1%self.average_slice_field_difference)*self.max_distance
            local_layer_height=Pa_Pb/number_of_segments
            return(calculate_ratios(local_layer_height,start_value,start_value+seg_max_dist))
    

    def find_thin_segment_groups(self):
        
        return create_connected_dicts(self.thin_segment)
    def get_group_root(self,group):

        return find_connections_and_min_width(self.segment,group)

    def apply_layer_path_inter_value(self,group,root):
        return calculate_p_values(group,root)

    def get_layer_path_scalar_field(self,layer,path):
        if layer>=0 :
            weight1=self.saddles_and_weights[layer]['weight']
        elif layer==-1 :
            key_rep=self.target_high.clustered_vkeys[path][0]
            weight1=self.mesh.vertex[key_rep]['scalar_field']
        elif layer==-2 :
            key_rep=self.target_low.clustered_vkeys[path][0]
            weight1=self.mesh.vertex[key_rep]['scalar_field']     
        else:
            print("bug: wrong layer")
        return weight1   
    def assign_segmentation_inter_values(self,inter_values):
        for seg_id,seg in enumerate(self.segment):
            layer1=self.segment[seg]['layers'][0]
            layer2=self.segment[seg]['layers'][1]
            offset=0.3
            offset*=self.average_slice_field_difference
            low_value=offset
            high_value=offset
            for layer_path in seg:
                if layer_path[0]==layer1:
                    if layer1 ==-2:
                        low_value=0.0001
                    elif layer_path in inter_values:
                        low_value=inter_values[layer_path]+offset
                elif layer_path[0]==layer2:
                    if layer2 ==-1:
                        high_value=1-0.0001
                    elif layer_path in inter_values:
                        high_value=inter_values[layer_path]+offset
            self.segment[seg]['inter_values']=(low_value/self.average_slice_field_difference,high_value/self.average_slice_field_difference)
            print('assign_segmentation_inter_values',seg_id,seg,low_value,high_value)
                    



def add_to_seg(segment,path_distance_field,connection,layer_path,dist_name):
    if layer_path not in segment.keys():
        segment[layer_path]={}
        segment[layer_path]['mask']=connection
    else:
        if segment[layer_path]['mask']!=connection:
            print("bug: one segment has two mask")                             
    segment[layer_path][dist_name]=path_distance_field      

def assign_scalar_field(mesh:Mesh,dist_high,dist_low,mask,output,boundary_seg):
    print('assign_scalar_field',len(dist_low),len(dist_high),len(mask))
    VN_old=min(len(dist_low),len(dist_high))
    if boundary_seg==0:
        for vertex in mesh.vertices():
            if mask[vertex]:
                mesh.vertex[vertex]['scalar_field']=dist_low[vertex]/(dist_low[vertex]+dist_high[vertex])
    elif boundary_seg==1:
        for vi,vertex in enumerate(mesh.vertices()):
            if mask[vertex]:
                if vertex<VN_old:
                    mesh.vertex[vertex]['scalar_field']=dist_low[vertex]/(dist_low[vertex]+dist_high[vertex])
                else:
                    mesh.vertex[vertex]['scalar_field']=0
    elif boundary_seg==2:
        #print(len(dist_low),len(dist_high))
        for vi,vertex in enumerate(mesh.vertices()):

            if mask[vertex]:
                if vertex<VN_old:
                    mesh.vertex[vertex]['scalar_field']=dist_low[vertex]/(dist_low[vertex]+dist_high[vertex])
                else:
                    mesh.vertex[vertex]['scalar_field']=1
    else:
        print("boundary_seg is not correct")
    g_evaluation = GradientEvaluation_Dart(mesh, output)
    g_evaluation.compute_gradient()
    g_evaluation.compute_gradient_norm()
    g_evaluation.compute_gradient_norm_max_neighbor_faces() 
    
    return g_evaluation      

def get_segment_mesh(mesh:Mesh,mask):
    new_mesh=copy.deepcopy(mesh)

    for vertex in mesh.vertices():
        if not mask[vertex]:
            
            for face in new_mesh.vertex_faces(vertex):
                new_mesh.delete_face(face)    
            new_mesh.delete_vertex(vertex)    
    



    return new_mesh
    


def List_in_List(list_a,list_b)->bool:
    '''
    
    ''' 
    #print(list_b[-1],"list_in_list")                          
    set_b = set(list_b) 

    for element in list_a:
        if element not in set_b :
            return False
    return True  


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
    
    return False
def expand_dict_values(d):
    updated = True
    while updated:  # Continue until no replacements are made
        updated = False
        for key, values in d.items():
            new_values = []
            for v in values:
                if v in d:  # If v is a key, replace it with its values
                    new_values.extend(d[v])
                    updated = True  # Mark that an update occurred
                else:
                    new_values.append(v)  # Keep the value as is
            d[key] = new_values  # Update the list for the current key
    return d
def create_connected_dicts(original_dict):
    # 创建一个无向图
    G = nx.Graph()

    # 将所有键添加到图中，并建立边
    keys = list(original_dict.keys())
    for i, tuple1 in enumerate(keys):
        tuple1_layer=[tup[0] for tup in tuple1]
        for j, tuple2 in enumerate(keys[i+1:], start=i+1):
            tuple2_layer=[tup[0] for tup in tuple2]
            if set(tuple1_layer) & set(tuple2_layer):  # 如果两个元组有交集
                G.add_edge(tuple1, tuple2)

    # 找出所有的连通分量
    connected_components = list(nx.connected_components(G))

    # 根据连通分量创建新的字典列表
    connected_dicts = []
    for component in connected_components:
        new_dict = {k: original_dict[k] for k in component if k in original_dict}
        connected_dicts.append(new_dict)

    return connected_dicts

def find_connections_and_min_width(A:Dict, B:Dict):
    # 提取键集合并构建字典C
    keys_A = set(A.keys())
    keys_B = set(B.keys())
    keys_C = keys_A - keys_B
    for key in keys_B:
        if key[-1][0]<-0.5:
            return key[-1]
    # 寻找C中与B相连的键
    connected_keys = set()  # 使用set来避免重复添加相同的键
    connections = {}  # 存储连接线条及其宽度

    for key_c in keys_C:
        for key_b in keys_B:
            common_elements = set(key_c) & set(key_b)
            if common_elements:
                connected_keys.add(key_c)
                connections[key_c] = common_elements.pop(), A[key_c]['width']

    # 找出最小宽度的键
    min_width_key = None
    min_width = float('inf')
   

    for key, (connection, width) in connections.items():
        if width < min_width:
            min_width = width
            min_width_key = key
         

    # 找到最小宽度键与B的所有连接线条
    all_connection_lines = []
    if min_width_key:
        for key_b in keys_B:
            common_elements = set(min_width_key) & set(key_b)
            if common_elements:
                all_connection_lines.extend(common_elements)

    return all_connection_lines

def  calculate_p_values(data_dict, root_tuple):
    if isinstance(root_tuple, List):
        root_tuple = root_tuple[0]
    print(data_dict.keys(),root_tuple)

    # 初始化P值字典，默认为None
    p_values = defaultdict(lambda: None)
    # 根节点的P值设为0
    p_values[root_tuple] = 0
    
    # 遍历队列，从根节点开始
    queue = deque([root_tuple])
    
    while queue:
        current_tuple = queue.popleft()
        #print(current_tuple)
        # 遍历所有的一级元组，寻找包含当前二级元组的那些
        

        for key in data_dict.keys():
            # print(key)
            if any(t[0] == current_tuple[0] for t in key):
                # 获取与当前二级元组在同一一级元组内的其他二级元组
                
                connected_tuples = [t for t in key if t != current_tuple]
                
                for connected_tuple in connected_tuples:
                    layer_info = data_dict[key]['layers']
                    width = data_dict[key]['width']
                    
                    # 确定P值调整量
                    if current_tuple[0] == layer_info[0]:
                        #print(current_tuple,layer_info,'up')
                        delta_p = width
                    elif current_tuple[0] == layer_info[1]:
                        #print(current_tuple,layer_info,'down')
                        delta_p = -width
                    else:
                        # 如果当前二级元组不在'layer'中，那么它应该与current_tuple绑定
                        print("Error: current_tuple not in layer_info",current_tuple,layer_info)
                        delta_p = 0
                    
                    # 更新相连的二级元组的P值
                    if p_values[connected_tuple] is None:
                        if connected_tuple[0]==current_tuple[0]:
                            p_values[connected_tuple] = p_values[current_tuple]
                        else:
                            p_values[connected_tuple] = p_values[current_tuple] + delta_p
                        #print("connected_tuple:",connected_tuple,p_values[connected_tuple],current_tuple,p_values[current_tuple])
                        queue.append(connected_tuple)
                        
                    # 确保绑定的二级元组有相同的P值
                    elif delta_p == 0 and p_values[connected_tuple] != p_values[current_tuple]:
                        raise ValueError("绑定的二级元组应具有相同的P值")
    
    return dict(p_values)

def split_interval(a, b, c, d, hp):
    """
    将区间 [a, b] 分割成宽度接近 hp 的小区间，其中第一个和最后一个小区间的宽度分别为 c * h0 和 d * h0。

    参数:
    a (float): 区间起点
    b (float): 区间终点
    c (float): 第一个小区间的宽度系数
    d (float): 最后一个小区间的宽度系数
    hp (float): 目标小区间宽度

    返回:
    h0 (float): 最优的小区间宽度
    intervals (list): 分割后的小区间列表，每个区间表示为 (start, end)
    """
    # 计算区间总长度
    L = b - a

    # 估算小区间数量 n
    n = round((L / hp) - c - d +2)

    diffmin=-1

    for ni in [n-1,n,n+1]:
        # 计算最优的 h0
        hc = L / (c + d + n-2 )
        diff=abs(hc-hp)
        if diffmin<0 or diff<diffmin:
            n=ni
            h0=hc

        

    # 生成小区间
    intervals = []
    current = a
    lenth=b-a

    # 第一个小区间
    first_width = c * h0
    current += first_width
    intervals.append(current/lenth)
    

    # 中间的小区间
    for _ in range(n - 2):
        current += h0
        if current>b:
            break
        intervals.append( current /lenth)
        



    return h0, intervals
def calculate_ratios(t, a, b):
    # 确保 a 小于 b
    if a >= b:
        print(t,a,b,'________bug_________')

        return []
    
    ratios = []
    current_point = math.ceil(a / t) * t
    
    # 如果起始点已经超过了b，则直接返回空列表
    if current_point > b:
        return ratios
    
    # 遍历从a到b之间每隔t取的点
    while current_point <= b:
        # 计算当前点的位置比例
        ratio = (current_point - a) / (b - a)
        ratios.append(ratio)
        
        # 移动到下一个点
        current_point += t
        
        # 如果下一个点超出了b，则停止循环
        if current_point > b:
            break
    
    return ratios


# def multi_source_dijkstra_path(mesh: Mesh, source:List[path_dart],search_direction=True):
#     """
#     mesh: compas mesh
#     source: List of path dart
#     search_direction: True or False if you want to search from the start or end of the edges intersecting the path
#     path_banned: List of path dart that you want to avoid during the sarching
#     """
    
#     # dict initialization, set infinity to every vertex
#     distances = {vertex: float("inf") for vertex in mesh.vertices()}

#     # set a queue to save the points and their distances
#     priority_queue = []

#     # for every source find the nearest verkey and set their diatance to the sources
  
#     source_points=[point for source_path in source for point in source_path.get_points()]
#     source_edges=[edge for source_path in source for edge in source_path.get_edges()]
#     for point,edge in zip(source_points,source_edges):
#         if search_direction:
#             flip=1
#         else:
#             flip=-1
#         edge_point1_scalar_field=flip*mesh.vertex_attribute(key=edge[0],name='scalar_field_org')
#         edge_point2_scalar_field=flip*mesh.vertex_attribute(key=edge[1],name='scalar_field_org')
#         if edge_point2_scalar_field>edge_point1_scalar_field:
#             edge_point=edge[1]
#         else:
#             edge_point=edge[0]
#         distance_source_edge_point=np.linalg.norm(np.array(point.data)-np.array(mesh.vertex_coordinates(edge_point)))
#         if distances[edge_point]>distance_source_edge_point:
#             distances[edge_point] = distance_source_edge_point
#             heapq.heappush(priority_queue, (0, edge_point))

#     while priority_queue:
#         current_distance, current_vertex = heapq.heappop(priority_queue)

#         # If the distance between the extracted vertices is greater than the known shortest distance, skip it
#         if current_distance > distances[current_vertex]:
#             continue
        

#         for neighbor in mesh.vertex_neighbors(current_vertex):
         

#             weigh = mesh.edge_length(current_vertex, neighbor)
#             distance = current_distance + weigh  

#             if distance < distances[neighbor]:
#                 distances[neighbor] = distance
#                 heapq.heappush(priority_queue, (distance, neighbor))
#     distance_list = list(distances.values())


   
#     topo_conection = [not math.isinf(d) for d in distance_list]
#     finite_count = sum(1 for d in topo_conection if d)

    
#     return distance_list,topo_conection
        # edges_cutting_data={}
        # faces_cutting_data={}
        # for layer_id in range(slicer.number_of_layers):
            
        #     layer=slicer.get_layer(layer_index=layer_id)
            # paths=layer.get_path()
            # up_or_down=saddles_and_weights[layer_id]['direction']
            
            
            # for path_id,path in enumerate(paths):
                
            #     for j,(edge,t) in enumerate(zip(path.get_edges(),path.get_t())):
            #         edge_flip=(edge[1],edge[0])
            #         if edge in edges_cutting_data:
            #             cut_t = edges_cutting_data[edge][1]
            #             if t > cut_t:
            #                 edge=(edges_cutting_data[edge][0],edge[1])
            #                 t=(t-cut_t)/(1-cut_t)
            #             else:
            #                 edge=(edge[0],edges_cutting_data[edge][0])
            #                 t=(t)/(cut_t) 
            #             edge_point_up=edge[1]
            #             edge_point_down=edge[0]   
            #             path.change_edge(j,edge,t)                        
            #         elif edge_flip in edges_cutting_data:
            #             cut_t = edges_cutting_data[edge_flip][1]
            #             if t > cut_t:
            #                 edge=(edges_cutting_data[edge_flip][0],edge[0])
            #                 t=(t-cut_t)/(1-cut_t)
            #             else:
            #                 edge=(edge[0],edges_cutting_data[edge_flip][0])
            #                 t=(t)/(cut_t) 
            #             edge_point_up=edge[1]
            #             edge_point_down=edge[0]
            #             path.change_edge(j,edge,t)
            #         else:
            #             edge_point1_scalar_field=self.mesh.vertex_attribute(key=edge[0],name='scalar_field_org')
            #             edge_point2_scalar_field=self.mesh.vertex_attribute(key=edge[1],name='scalar_field_org')
            #             if edge_point2_scalar_field>edge_point1_scalar_field:
            #                 edge_point_up=edge[1]
            #                 edge_point_down=edge[0]
            #             else:
            #                 edge_point_up=edge[0]
            #                 edge_point_down=edge[1]
                    
            #         org_faces=new_mesh.edge_faces(u=edge[0],v=edge[1])
            #         #print(org_faces)
            #         org_faces_vertices=[]
            #         for face in org_faces:
            #             vertices_face=copy.deepcopy(new_mesh.face_vertices(face))
            #             vertices_face.remove(edge[0])
            #             vertices_face.remove(edge[1])
            #             org_faces_vertices.append(vertices_face[0])
                    
            #         #print(org_faces_vertices)
            #         try:
            #             w=trimesh_split_edge(mesh=new_mesh,u=edge[0],v=edge[1],t=t)
            #         except:
            #             print('bug',layer_id,path_id,edge,j,[set(cutted_edge) for cutted_edge in edges_cutting_data.keys()])
                    
            #         new_faces=new_mesh.vertex_faces(w)
            #         for face,face_vertex in zip(org_faces,org_faces_vertices):
            #             faces_cutting_data[face]=[]
            #             for new_face in new_faces:
                           
            #                 if face_vertex in new_mesh.face_vertices(new_face):
                                
            #                     faces_cutting_data[face].append(new_face)
            #         #print((edge_point_down,edge_point_up),w,t)
            #         edges_cutting_data[(edge_point_down,edge_point_up)]=(w,t)
                    

            #         path.apply_w_key(w,j)
            #         path.add_edge_up_down(j,edge_point_up,edge_point_down)
            # for path in layer.get_path():
            #     for point_id,edge in enumerate(path.get_edges()):
            #         w,edge_point_up
            #         faces_up=new_mesh.edge_faces(u=path.get_edge_up_pt(point_id),v=path.get_keys()[point_id])
            #         faces_down=new_mesh.edge_faces(u=path.get_edge_down_pt(point_id),v=path.get_keys()[point_id])
            #         path.add_faces(faces_up=faces_up,faces_down=faces_down)
              
        #print(faces_cutting_data)
if __name__ == "__main__":
    main()
    


    




    
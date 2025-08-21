from compas.datastructures import Mesh
from compas_slicer.pre_processing import CompoundTarget
import scipy.stats
from interpolationdart import DartPreprocesssor
import os
from compas.datastructures import Mesh
import logging
import compas_slicer.utilities as utils
from compas_slicer.post_processing import simplify_paths_rdp
from compas_slicer.print_organization import InterpolationPrintOrganizer
from compas_slicer.post_processing import seams_smooth
from compas_slicer.print_organization import smooth_printpoints_up_vectors, smooth_printpoints_layer_heights
import time
from interpolationdart import DartPreprocesssor
#from mesh_changing import change_mesh_shape
import progressbar
import numpy as np
import igl
from compound_target_dart import save_nested_list,load_nested_list,mesh_smooth_centroid_vertices_influence
from interpolation_slicer_dart import InterpolationSlicer_Dart
from compas_slicer.pre_processing.preprocessing_utils.geodesics import get_igl_EXACT_geodesic_distances
from compound_target_dart import CompoundTargetDart
import scipy
import math
import copy
import time
from distances import get_real_distances,print_list_with_details,cube_distances,get_close_mesh,cube_distances_multi_sources

class optimizing_offset_value:
    def __init__(self,mesh:Mesh,saddles_ge,target01:CompoundTargetDart,target02:CompoundTargetDart,processor:DartPreprocesssor,Outputpath,sinha,animation_frame='',only_normalize_big_data=False,saddle_sides_limit=None,single_side=False) -> None:
        self.saddle_sides_limit=saddle_sides_limit
        self.mesh=mesh
        self.saddles=saddles_ge.saddles
        self.target_low=target01
        self.target_high=target02
        self.preprocessor = processor
        self.target_high
        self.Outputpath=Outputpath
        self.VN = len(list(self.mesh.vertices()))
        self.number_of_saddles=len(list(self.saddles))
        self.distances_from_saddles=[]
        self.distances_from_saddles_flipped=[]
        self.sinha=sinha

        self.saddles_facing_High=[]
        self.saddles_facing_Low=[]
        self.saddles_facing_High_vertices=[]
        self.saddles_facing_Low_vertices=[]

        self.get_saddles_direction()
        self.influence_lists_High=[]
        self.influence_lists_Low=[]
        self.influence_lists_target_High=[]
        self.influence_lists_target_Low=[]
        self.distances_from_target_high_flipped=[]
        self.distances_from_target_low_flipped=[]
        self.distances_from_target_high=[]
        self.distances_from_target_low=[]
        
        self.number_of_saddles_targets=self.number_of_saddles+self.target_high.number_of_boundaries+self.target_low.number_of_boundaries
        self.influence_manual_list=[0]*self.number_of_saddles_targets
        self.high_list=[0]*self.number_of_saddles_targets
        self.items_target_high=list(range(self.number_of_saddles,self.number_of_saddles+self.target_high.number_of_boundaries))
        self.items_target_low=list(range(self.number_of_saddles+self.target_high.number_of_boundaries,self.number_of_saddles_targets))
        self.items_saddles=list(range(self.number_of_saddles))
        self.animation_frame=animation_frame
        self.only_normalize_big_data=only_normalize_big_data
        self.single_side=single_side
        # #whole
        # self.high_list=[12,102,152,62,102,62,12,152,
        #                 0,62,
        #                 202,132]
        
        # Jun bg
        # self.high_list=[0,100,100,100,150,0,
        #                 0,0,
        #                 100,150]  
        # self.influence_manual_list=[20,20,20,20,20,20,
        #                             50,50,
        #                             50,50]   
        # JUl ah
        # self.high_list=[850,650,450,300,370,500,
        #                 0,300,
        #                 900,1200]
        # self.influence_lists=[0.7,1,1,0.5,0.7,0.7,
        #                  0.4,0.7,
        #                  1,1,]
        #Jul ai1
        # self.high_list=[220,250,200,370,350,100,170,
        #                 0,300,
        #                 0,0,0]
        # self.influence_manual_list=[250,[250,0.7],150,100,200,[0,0.5],[100,0.7],
        #                   [0,0.4],[100,0.8],
        #                   [0,0.3],0,200]  
        #Jul ai2  
        # self.high_list=[150,70,80,80,
        #                 160,70,0,0,
        #                 80,0,
        #                 150,150,]
        # self.influence_manual_list=[300,300,300,300,
        #                             300,300,300,300,
        #                   300,300,
        #                   300,300] 
        # self.influence_manual_list=[500,500,500,500,
        #                             500,500,500,500,
        #                   500,500,
        #                   500,500] 
        # Jul_ba
        # self.high_list=[100,250,100,200,
        #                 200,100,250,250,
        #                 100,50,180,
        #                 0,0,0,
        #                 200,400,]
        # self.influence_manual_list=[100,100,0,0,
        #                             0,0,100,100,
        #                             100,100,150,
        #                             50,50,50,
        #                             0,0]  
        # table-1
        # self.high_list=[100,
        #                 80,0,30,100,30,
        #                 0,0,70,70,100,
        #                 0,0,0,
        #                 100,100,]
        # self.influence_manual_list=[100,
        #                             100,150,100,100,100,
        #                             100,150,100,100,100,
        #                             100,100,100,
        #                             100,100,]
        # # table_2
        # self.high_list=[0,0,0,30,
        #                 50,50,60,60,100,
        #                 130,150,
        #                 0,0,0,
        #                 170,170,170]      
        # self.influence_manual_list=[100,
        #                             100,200,200,100,100,
        #                             100,100,100,100,300,
        #                             300,300,300,
        #                             300,300,300,] 
        #Aug bg
        # self.high_list=[100,0,0,0,100,100]  
        # csch
        # self.high_list=[ 100,0,200,100,
        #                 0,0,
        #                 100,200]
        #csch2
        # self.high_list=[2,0,2,1,
        #                 2,0,1,0]
        # self.influence_manual_list=[18,18,0,0,
        #                             0,0,0,0]
        # beam 3B
        # self.high_list=[0,41,41,31,0,
        #                 0,0,14,0,0,0,0]
        # BEAM C
        # self.influence_manual_list=[70,70,70,70,
        #                              70,70,70,70,
        #                              70,70,70,70,
        #                              70,70,70,70,
        #                              70,70,70,70,
        #                              70,70,70,70,
        #                              70,70,70,70,]
        # only target
        # self.influence_manual_list=[-2,-2,-2,-2,
        #                                 -2,-2,-2,-2,
        #                                 0,0,0,0]
        # only saddles
        # self.influence_manual_list=[0,0,0,0,
        #                                 0,0,0,0,
        #                                 -2,-2,-2,-2]
        # table_2
        self.high_list=[20,20,0,0,
                        50,50,30,30,
                        0,0,
                        50,50]      
        # self.influence_manual_list=[100,100,100,100,100,100,
        #                             100,100,
        #                             100,100,
        #                             100,100] 
    def analysis_gradient(self):
        self.gradients=self.preprocessor.create_all_distance_fields_evalueation(target_1=self.target_low, target_2=self.target_high, save_output=True) 
        print("gradients",self.gradients)       
    def get_saddles_direction(self):
        """
        将鞍点分为朝上与朝下的
        """
        for i,saddle in enumerate(self.saddles):
            normal=self.mesh.vertex_normal(key=saddle)[2]
            if normal>0:
                self.saddles_facing_High.append(i)
                self.saddles_facing_High_vertices.append(saddle)
            else:
                self.saddles_facing_Low.append(i)
                self.saddles_facing_Low_vertices.append(saddle)
    # def creat_cutting_lines(self):
    #     with progressbar.ProgressBar(max_value=len(self.saddles)) as bar:
    #         for i,saddle in enumerate(self.saddles):
    #             print("generate cutting line for saddle",i)
    #             self.creat_cutting_line(saddle)
    #             bar.update(i)  # advance progress bar
    
    # def creat_cutting_line(self,saddle):       
    #     saddle_weigh=self.preprocessor.saddle_creat_compound_target_and_create_gradient_evaluation(saddle=saddle)
    #     slicer = InterpolationSlicer_Dart(mesh=self.mesh, preprocessor=self.preprocessor)
    #     slicer.slice_model_by_weigh(weigh=saddle_weigh)  # compute_norm_of_gradient contours
    #     simplify_paths_rdp(slicer, threshold=0.25)
    #     seams_smooth(slicer, smooth_distance=3)
    #     slicer.printout_info()
    #     filename=f"{saddle}{'curved_slicer.json'}"
    #     print (filename)
    #     utils.save_to_json(slicer.to_data(), self.Outputpath, filename)

        
    
    def get_selected_faces(self):
        pass

    def get_new_mesh(self,face_items,vertices):
        return self.mesh.from_vertices_and_faces(faces=face_items,vertices=vertices)
    
    def offset_target_only_related_boundary(self):
        self.target_high.get_saddles_number(self.number_of_saddles)
        self.target_low.get_saddles_number(self.number_of_saddles)
        self.creat_saddles_target_influence_field(method='igl')
        self.creat_sources_offset_distances()

    
    
    
    def offset_target_base_on_spatail_interpolation(self):
        # 区分鞍点朝向
        self.target_high.get_saddles_number(self.number_of_saddles)
        self.target_low.get_saddles_number(self.number_of_saddles)
        


        # self.get_close_mesh()
        if False:
            sinha_org=self.sinha
            influence_list_org=self.influence_manual_list.copy()
            self.influence_manual_list=[-2,-2,-2,-2,
                                        -2,-2,-2,-2,
                                        0,0,0,0]
            self.sinha=-0.5
            self.creat_saddles_target_influence_field(method='igl')
            self.creat_sources_offset_distances()
            self.sinha=sinha_org
            self.influence_manual_list=influence_list_org
            self.influence_lists=self.get_saddle_target_influence_field('influences')
        elif self.only_normalize_big_data:
            self.only_normalize_big_data=False
            self.creat_saddles_target_influence_field(method='igl')
            self.creat_sources_offset_distances()
            self.only_normalize_big_data=True
            self.influence_lists=self.get_saddle_target_influence_field('influences')
        else:
            self.creat_saddles_target_influence_field(method='igl')
            self.creat_sources_offset_distances()
        
        
        self.target_high.offset_distance_with_saddles_targets(saddles_targets_offset_distances_list=[
                                                                                                    self.saddles_offset_lists_High_Whole,
                                                                                                    self.High_targets_offset_lists_High,
                                                                                                    self.High_targets_offset_lists_Low
                                                                                                    ],
                                                                                                    other_target_self_offset_distance_list=self.Low_targets_offset_lists_High,
                                                                                                    frame=self.animation_frame,
                                                                                                    single_side=self.single_side)
        self.target_low.offset_distance_with_saddles_targets(saddles_targets_offset_distances_list=[
                                                                                                    self.saddles_offset_lists_Low_Whole,
                                                                                                    self.Low_targets_offset_lists_Low,
                                                                                                    self.Low_targets_offset_lists_High
                                                                                                    ],
                                                                                                    other_target_self_offset_distance_list=self.High_targets_offset_lists_Low,
                                                                                                    frame=self.animation_frame,
                                                                                                    single_side=self.single_side)
        #self.saddle_modify()
    #def saddle_modify(self,saddle,up_b_saddles,low_b_saddles)

    def get_close_mesh(self):
        self.close_mesh=get_close_mesh(self.mesh,self.target_high.clustered_vkeys,self.target_low.clustered_vkeys)
      

    # def creat_saddle_offset_distances(self):
    #     self.pare_saddles_with_target()
    #     self.pare_saddles() 
    #     self.pare_targets_with_saddles()
    #     self.creat_saddles_graph()
    #     if False:
    #         self.saddles_offset_lists_High_Whole=self.creat_saddle_offset_distances_one_side(self.saddles_facing_High,self.target_high)
    #         self.saddles_offset_lists_Low_Whole=self.creat_saddle_offset_distances_one_side(self.saddles_facing_Low,self.target_low)
    #         self.creat_saddle_offset_distances_one_side_again(self.saddles_offset_lists_High_Whole)
    #         self.creat_saddle_offset_distances_one_side_again(self.saddles_offset_lists_Low_Whole)
    #     else:
    #         self.saddles_offset_lists_High_Whole=[[0.0, 62.65426618497253],[0.0, 62.65426618497253],[0.0, 62.65426618497253],[0.0, 62.65426618497253],
    #                                               [0.0, 62.65426618497253],[0.0, 62.65426618497253],[0.0, 62.65426618497253],[0.0, 62.65426618497253],]
    #         self.saddles_offset_lists_Low_Whole=[[0.0, 51.89041869134394],[0.0, 51.89041869134394],[0.0, 51.89041869134394],[0.0, 51.89041869134394],
    #                                               [0.0, 51.89041869134394],[0.0, 51.89041869134394],[0.0, 51.89041869134394],[0.0, 51.89041869134394],]
        
        
    #     # print_list_with_details(self.saddles_offset_lists_High_Whole,"saddles_offset_lists_High_Whole")
    #     # print_list_with_details(self.saddles_offset_lists_Low_Whole,"saddles_offset_lists_Low_Whole")

    def creat_sources_offset_distances(self):
        if self.number_of_saddles==1:
            self.saddles_offset_lists_High_Whole=self.creat_saddle_offset_distances_one_side(self.saddles_facing_High,self.target_high)
            self.saddles_offset_lists_Low_Whole=self.creat_saddle_offset_distances_one_side(self.saddles_facing_Low,self.target_low)
            if self.saddles_facing_High:
                print("saddle facing high",self.saddles_facing_High)
            elif self.saddles_facing_Low:
                print("saddle facing low",self.saddles_facing_Low)
        else:

            self.pare_saddles_with_target()
            self.pare_saddles() 
            self.pare_targets_with_saddles()
            
            if True:
                self.saddles_offset_lists_High_Whole=self.creat_saddle_offset_distances_one_side(self.saddles_facing_High,self.target_high)
                self.saddles_offset_lists_Low_Whole=self.creat_saddle_offset_distances_one_side(self.saddles_facing_Low,self.target_low)


                #self.creat_saddles_graph()



            
                print("creat_saddle_offset_distances_one_side_again")
                self.creat_saddle_offset_distances_one_side_again(self.saddles_offset_lists_High_Whole)
                self.creat_saddle_offset_distances_one_side_again(self.saddles_offset_lists_Low_Whole)
            elif True:# only saddle
               
                self.saddles_offset_lists_High_Whole=[[1000, 1137] for _ in range(8)]
                self.saddles_offset_lists_Low_Whole=[[1126.0, 1100] for _ in range(8)]
            elif True:
                self.saddles_offset_lists_High_Whole=[[0.0,137.450957840837651] for _ in range(8)]
                self.saddles_offset_lists_Low_Whole=[[0,94.60886264254359] for _ in range(8)]


            list_high=self.high_list
            print("saddles_offset_lists_High_Whole")
            for id,offset_list in enumerate(self.saddles_offset_lists_High_Whole):
                print("saddle:",id,offset_list)
            print('saddles_offset_lists_Low_Whole')
            for id,offset_list in enumerate(self.saddles_offset_lists_Low_Whole):
                print("saddle:",id,offset_list)



            def replace_min_with_zero(matrix):
                if not matrix or not matrix[0]:  # 如果矩阵为空，直接返回
                    return
                
                min_value = float('inf')
                min_index = (-1, -1)
                print(matrix)
                # 找到最小值的位置
                for i in range(len(matrix)):
                    if matrix[i][i] < min_value:
                        min_value = matrix[i][i]
                        min_index = (i, i)
                
                # 将最小值替换为0
                i, j = min_index
                if i != -1 and j != -1:
                    matrix[i][j] = 0
            self.High_targets_offset_lists_High=self.creat_boundary_curves_offset_value(source=True,boundary=True)
            print('High_source_offset_lists_on_high_boundary')
            for id,offset_list in enumerate(self.High_targets_offset_lists_High):
                print("target:",id,offset_list)            
            self.High_targets_offset_lists_Low=self.creat_boundary_curves_offset_value(source=False,boundary=True)
            print('Low_source_offset_lists_on_high_boundary')
            for id,offset_list in enumerate(self.High_targets_offset_lists_Low):
                print("target:",id,offset_list) 
            self.Low_targets_offset_lists_High=self.creat_boundary_curves_offset_value(source=True,boundary=False)
            print('High_source_offset_lists_on_low_boundary')
            for id,offset_list in enumerate(self.Low_targets_offset_lists_High):
                print("target:",id,offset_list) 
            self.Low_targets_offset_lists_Low=self.creat_boundary_curves_offset_value(source=False,boundary=False)
            print('Low_source_offset_lists_on_low_boundary')
            for id,offset_list in enumerate(self.Low_targets_offset_lists_Low):
                print("target:",id,offset_list) 
            # self.High_targets_offset_lists_High = [copy.deepcopy(self.saddles_offset_lists_High_Whole[saddle])  for saddle in self.target_high.pare_saddles ]
            # #replace_min_with_zero(self.High_targets_offset_lists_High)
            # self.High_targets_offset_lists_Low =[copy.deepcopy(self.saddles_offset_lists_High_Whole[saddle])  for saddle in self.target_low.pare_saddles] 
            # self.Low_targets_offset_lists_High = [copy.deepcopy(self.saddles_offset_lists_Low_Whole[saddle])  for saddle in self.target_high.pare_saddles ]
            
            # self.Low_targets_offset_lists_Low = [copy.deepcopy(self.saddles_offset_lists_Low_Whole[saddle])  for saddle in self.target_low.pare_saddles ] 
            #replace_min_with_zero(self.Low_targets_offset_lists_Low)
            print(self.High_targets_offset_lists_High,self.High_targets_offset_lists_Low,self.Low_targets_offset_lists_High,self.Low_targets_offset_lists_Low)


            try:
                print("/////////manual offsetting ///////")
                max_high=max(list_high)
                list_low=[max_high-x for x in list_high]

                print(self.number_of_saddles,(self.number_of_saddles+self.target_high.number_of_boundaries),list_high[self.number_of_saddles:(self.number_of_saddles+self.target_high.number_of_boundaries)],list_high[(self.number_of_saddles+self.target_high.number_of_boundaries):])
                print("S_H")
                self.manul_add(self.saddles_offset_lists_High_Whole,list_high[:self.number_of_saddles])
                print("S_L")
                self.manul_add(self.saddles_offset_lists_Low_Whole,list_low[:self.number_of_saddles])
                print("H_H")
                self.manul_add(self.High_targets_offset_lists_High,list_high[self.number_of_saddles:(self.number_of_saddles+self.target_high.number_of_boundaries)])
                print("H_L")
                self.manul_add(self.High_targets_offset_lists_Low,list_high[(self.number_of_saddles+self.target_high.number_of_boundaries):])
                print("L_H")
                self.manul_add(self.Low_targets_offset_lists_High,list_low[self.number_of_saddles:(self.number_of_saddles+self.target_high.number_of_boundaries)])
                print("L_L")
                self.manul_add(self.Low_targets_offset_lists_Low,list_low[(self.number_of_saddles+self.target_high.number_of_boundaries):])
                
            except:
                print("no manul list")
            print("saddles_offset_lists_High_Whole_manul")
            for id,offset_list in enumerate(self.saddles_offset_lists_High_Whole):
                print("saddle:",id,offset_list)
            print('saddles_offset_lists_Low_Whole_manul')
            for id,offset_list in enumerate(self.saddles_offset_lists_Low_Whole):
                print("saddle:",id,offset_list)
    def creat_boundary_curves_offset_value(self,source,boundary):
        '''
        source:
        True Targer Higj
        Low Target LOw
        boundary:
        True High
        Low Low
        '''
        if boundary:
            offset_value_list=[0]*self.target_high.number_of_boundaries
            saddles_offset_list=self.saddles_offset_lists_High_Whole
            
        else:
            offset_value_list=[0]*self.target_low.number_of_boundaries
            saddles_offset_list=self.saddles_offset_lists_Low_Whole
            
        offset_value_lists=[]
        if source:
            for i in range(self.target_high.number_of_boundaries):
                offset_value_lists.append(offset_value_list.copy())
            
            sourse_target=self.target_high
        else:
            for i in range(self.target_low.number_of_boundaries):
                offset_value_lists.append(offset_value_list.copy())
            
            sourse_target=self.target_low
         

        for i_source,target_source in enumerate(offset_value_lists):
            for i_target,target_bounary in enumerate(target_source):
                target_offset_value=0
                None_inflence=0
                for i_saddle,inflence in enumerate(sourse_target.pare_saddles[i_source]):
                    saddle_offset_value=saddles_offset_list[i_saddle][i_target]
                    #print(saddle_offset_value,inflence)
                    if saddle_offset_value is not None:
                        target_offset_value+=saddle_offset_value*inflence
                    else:
                        None_inflence+=inflence
                if None_inflence>0.7:
                    target_source[i_target]=None
                else:
                    target_source[i_target]=target_offset_value/(1-None_inflence)
                
        return offset_value_lists


        # for i,saddle in enumerate(self.saddles):#对于面向下的鞍点，其上侧边缘的偏移值为0，要通过周围面向上的点来补齐，反之亦然。
        #     if saddles_offset_lists[i] == None:
        #         start=False
                
        #         for j,saddle in enumerate(self.saddles):
        #             if saddles_offset_lists[j] is not None:#遍历所有其它的点，如果不是同侧的，则根据影响力补全偏移距离表
        #                 if not start:#初始化
        #                     start=True

        #                     saddles_offset_list=[0]*len(saddles_offset_lists[j])
        #                     saddles_None_list=[0]*len(saddles_offset_lists[j])
        #                     for k,offset_dist in enumerate(saddles_offset_lists[j]):
        #                         if offset_dist is None:
        #                             saddles_None_list[k]=self.saddles_pare_saddles[i][j]
        #                         else:
        #                             saddles_offset_list[k]=offset_dist*self.saddles_pare_saddles[i][j]
        #                     # print_list_with_details(saddles_offset_list,"saddles_offset_list")
        #                     # saddles_offset_list=np.array([offset_dist*self.saddles_pare_saddles[i][j] for offset_dist in saddles_offset_lists[j] ])
        #                 else:
        #                     for k,offset_dist in enumerate(saddles_offset_lists[j]):
        #                         if offset_dist is None:
        #                             saddles_None_list[k]+=self.saddles_pare_saddles[i][j]
        #                         else:
        #                             #print(saddles_offset_list[k],offset_dist,self.saddles_pare_saddles[i][j],k)
        #                             saddles_offset_list[k]+=offset_dist*self.saddles_pare_saddles[i][j]                            
        #         for k,offset_none in enumerate(saddles_None_list):
        #             if offset_none>0.9:
        #                 saddles_offset_list[k]=None
        #             else:
        #                 saddles_offset_list[k]/=(1-offset_none)


        #         saddles_offset_lists[i]=saddles_offset_list
        # #print_list_with_details(saddles_offset_lists,"saddles_offset_lists")

    def manul_add(self,list_org,add_list):
        for i,num in enumerate(add_list):
            for j,x in enumerate(list_org[i]):
                
                if isinstance(x,list):
                    print(x,"has already manual offset, bug_______________________________")
                    list_org[i][j][1]=x[1]+num
                    print(list_org[i][j],i,j)
                elif x is not None:
                    #print(x)
                    list_org[i][j]=x+num    
                elif x is None:
                    
                    list_org[i][j]=[None,num]
                    print("manul_add manual None,",'saddle_point:',i,'boundary:',j,'list after add',list_org[i][j])
                    
                else:
                    print("manul add bug??????????????????")

        print(list_org,"manul add slist")

    def creat_saddles_graph(self):
        self.get_saddle_target_distances(method='in_mesh')
        self.near_field()
        self.saddle_graph=self.get_graph() 
             
        print_list_with_details(self.saddle_graph,"saddle_graph")

        self.saddles_neibor_saddles=self.build_adjacency_list(self.saddle_graph,(self.number_of_saddles+self.target_high.number_of_boundaries+self.target_low.number_of_boundaries))
        self.get_graph_derection()  
        save_nested_list(nested_list=self.saddle_graph,file_name="graph.json",file_path=self.Outputpath)
        print(self.saddles_neibor_saddles,"saddles_neibor_saddles")

    def get_graph_derection(self):
        saddle_edges_derection=[None]*len(self.saddle_graph)
        for i,edge in enumerate(self.saddle_graph):
            start=edge[0]
            end=edge[1]
            
            if start in self.items_target_high:
                if end in self.items_target_high:
                    print(start,end,"bug with target high neibor target high, one saddle point missing")
                elif end in self.items_target_low or end in self.items_saddles:
                    print(start,end,"False target")
                    saddle_edges_derection[i]=False
                else :
                    print(start,end,"!!!!!!!!!!!!!!!!!!!!!!bug")
            elif start in self.items_target_low:
                if end in self.items_target_low:
                    print(start,end,"bug with target low neibor target low, one saddle point missing")
                elif end in self.items_target_high or end in self.items_saddles:
                    print(start,end,"True target")
                    saddle_edges_derection[i]=True
                else:
                    print(start,end,"!!!!!!!!!!bug!!!!!!!!!")
            else:
                if end in self.items_target_high:
                    saddle_edges_derection[i]=True
                    print(start,end,"True target")
                elif end in self.items_target_low:
                    saddle_edges_derection[i]=False
                    print(start,end,"False target")  
                else:                                  
                    #elif end in self.items_target_high:



                    neibor_start=self.saddles_neibor_saddles[start]
                    neibor_end=self.saddles_neibor_saddles[end] 
                    duplicates = list(set(neibor_start) & set(neibor_end))
                    derection=self.FH_FL_derection(start,end)
                    if isinstance(derection,int):
                        derection= self.FH_FL_derection(end,start)
                        if not isinstance(derection,int):
                            if isinstance(derection,list):
                                derection=[not d for d in derection]
                            else:
                                derection=not derection
                    saddle_edges_derection[i]=derection

                    if derection is True or derection is False or derection is None or isinstance(derection,list):
                        print(start,end,derection)
                    else:
                        self.F_same_derection(start,end)
                        print(start,end,duplicates,derection)
    def FH_FL_derection(self,start,end):
        def add_none(a,b):
            if a is None or b is None:
                return(np.Infinity)
            else:
                return(a+b)
        if start in self.saddles_facing_High:
            high_offset1=self.saddles_offset_lists_High_Whole[start]
            high_distance_list_start=self.target_high.get_all_distances_for_vkey(self.saddles[start])
            high_distance_list_start_offset=[add_none(a,b) for a,b in zip(high_distance_list_start,high_offset1)]
            low_distance_list_start=self.target_low.get_all_distances_for_vkey(self.saddles[start])
            if end in self.saddles_facing_Low:
                low_offset2=self.saddles_offset_lists_Low_Whole[end]
                low_distance_list_start_offset=[add_none(a,b) for a,b in zip(low_distance_list_start,low_offset2)]
                min_start_distance_low=min(low_distance_list_start_offset)
                
                cloest_target_low=low_distance_list_start_offset.index(min_start_distance_low)
                low_distance_list_end=self.target_low.get_all_distances_for_vkey(self.saddles[end])
                low_distance_list_end_offset=[add_none(a,b) for a,b in zip(low_distance_list_end,low_offset2)]
                min_end_distance_low=low_distance_list_end_offset[cloest_target_low]

                high_distance_list_end=self.target_high.get_all_distances_for_vkey(self.saddles[end])
                high_distance_list_end_offset=[add_none(a,b) for a,b in zip(high_distance_list_end,high_offset1)] 
                min_end_distance_high=min(high_distance_list_end_offset)

                cloest_target_high=high_distance_list_end_offset.index(min_end_distance_high)
                min_start_distance_high=high_distance_list_start_offset[cloest_target_high]
                #print(start,end)
                if min_end_distance_low>min_start_distance_low and min_start_distance_high>min_end_distance_high:
                    #print("good1",start,end)
                    return(True)
                elif min_end_distance_low<min_start_distance_low and min_start_distance_high<min_end_distance_high:
                    #print("good2",start,end)
                    return(False)
                elif min_end_distance_low>min_start_distance_low and min_start_distance_high<min_end_distance_high:
                    #print("bad",start,end)  
                    return([False,True])
                elif min_end_distance_low<min_start_distance_low and min_start_distance_high>min_end_distance_high:
                    #print("bad",start,end) 
                    return([True,False])
                else:
                    print("======")
            else:
                #print("pass",start,end)
                return(3)
        else:
            #print("try mirror")
            return(3)
    def F_same_derection(self,start,end):
        if start  in self.saddles_facing_High and end in self.saddles_facing_High:
            target=self.target_high
            SO_distance_list=self.saddles_offset_lists_High_Whole
        elif start  in self.saddles_facing_Low and end in self.saddles_facing_Low:
            target=self.target_low
            SO_distance_list=self.saddles_offset_lists_Low_Whole
        else:
            print(start,end,"not same derection")
            return(3)
        print(start,end,SO_distance_list[start],SO_distance_list[end])
        

        





    def near_field(self):
        """
        get weight field with nearest neighbor interpolation
        """
        self.near_field_list=[]
        for vertex in self.mesh.vertices():
            infs=[]
            for i in range(self.number_of_saddles+self.target_high.number_of_boundaries+self.target_low.number_of_boundaries):
                inf=self.distances_from_saddles_and_target[vertex][i]

                infs.append(inf)
            min_saddle_target=infs.index(min(infs))
            self.near_field_list.append(min_saddle_target) 
        save_nested_list(nested_list=self.near_field_list,file_name="near_field.json",file_path=self.Outputpath) 

    def get_graph(self,min_connections=2):

        # 使用一个字典来存储每对相连片之间的连接数量
        connection_counts = {}

        # 遍历邻接列表，检查每对相邻点的片号
        for vi in self.mesh.vertices():
            cluster1 = self.near_field_list[vi]
            for neighbor in self.mesh.vertex_neighbors(vi):
                cluster2 = self.near_field_list[neighbor]
                if cluster1 != cluster2:
                    # 记录不同片之间的连接关系
                    pair = tuple(sorted((cluster1, cluster2)))
                    if pair not in connection_counts:
                        connection_counts[pair] = 0
                    connection_counts[pair] += 1

        # 使用一个集合来存储符合条件的连接
        connections = {pair for pair, count in connection_counts.items() if count > min_connections}

        return list(connections)
    
    def build_adjacency_list(self,connections, num_points):
        # 初始化邻接列表，每个点的邻居列表开始都是空的
        #print(num_points)
        adjacency_list = [[] for _ in range(num_points)]

        # 遍历连接关系列表，填充邻接列表
        for p1, p2 in connections:
            adjacency_list[p1].append(p2)
            #print(p2)
            adjacency_list[p2].append(p1)

        return adjacency_list

     

    
                
            





    def creat_saddle_offset_distances_one_side(self,saddle_items,target:CompoundTargetDart):
        """
        get the offset value for nested vertices for one compoundtargt
        """
        lists_from_target=[]
        for i,saddle in enumerate(self.saddles):
            if i in saddle_items:
                lists_from_target.append(target.get_offset_distances_saddle(saddle=saddle,sides_limitation=self.saddle_sides_limit))
            else:
                lists_from_target.append(None)
        
        print("lists_from_target",lists_from_target)
        return (lists_from_target)

    def pare_saddles_with_target(self):
        self.saddles_pair_targets=[0]*self.number_of_saddles
        for i,saddle in enumerate(self.saddles):
            distances=[]
            if i in self.saddles_facing_High:
                for j in range(self.target_low.number_of_boundaries):
                    distances.append(self.get_distance_between_one_saddle_and_target(i,self.target_low,j))
                pare_target=distances.index(min(distances))
                self.saddles_pair_targets[i]=pare_target                    
            else:
                for j in range(self.target_high.number_of_boundaries):
                    distances.append(self.get_distance_between_one_saddle_and_target(i,self.target_high,j))
                pare_target=distances.index(min(distances))
                self.saddles_pair_targets[i]=pare_target
        print(self.saddles_pair_targets,"self.saddles_pair_targets")


    # def pare_saddles_targets_with_saddles_targets(self):

    #     self.saddles_targets_pare_saddles_targets=[0]*self.number_of_saddles_targets
    #     for i in range(self.number_of_saddles_targets):
    #         distances=[]
    #         distances_p=[] 
    #         if i in self.saddles_facing_High or (i >= self.number_of_saddles and i < self.number_of_saddles+self.target_high.number_of_boundaries):     
    #             for j in range(self.number_of_saddles_targets): 
    #                 if j in self.saddles_facing_Low or (j >= self.number_of_saddles+self.target_high.number_of_boundaries):
    #                     distance=(self.get_distance_between_saddles_targets(i,j))
    #                     if j <self.number_of_saddles and i < self.number_of_saddles:
    #                         if self.get_distance_between_one_saddle_and_target(i,self.target_low,self.saddles_pair_targets[i]) > self.get_distance_between_one_saddle_and_target(j,self.target_low,self.saddles_pair_targets[i]):
    #                             print("distance to low ok",i,self.saddles_pair_targets[i],j)
    #                             distances.append(distance) 
    #                         else:
    #                             distances.append(math.inf)
    #                     else:
    #                         distances.append(distance)
    #                 else:
    #                     distances.append(math.inf)
    #             pare_saddle_target=distances.index(min(distances))
    #         else:
    #             for j in range(self.number_of_saddles_targets): 
    #                 if j in self.saddles_facing_High or (j >= self.number_of_saddles and j < self.number_of_saddles+self.target_high.number_of_boundaries):
    #                     distance=self.get_distance_between_saddles_targets(i,j)
    #                     if j < self.number_of_saddles and i < self.number_of_saddles:
    #                         if self.get_distance_between_one_saddle_and_target(i,self.target_high,self.saddles_pair_targets[i]) > self.get_distance_between_one_saddle_and_target(j,self.target_high,self.saddles_pair_targets[i]):
    #                             print("distance to high ok",i,self.saddles_pair_targets[i],j)
    #                             distances.append(distance) 
    #                         else:
    #                             distances.append(math.inf)
    #                     else:
    #                         distances.append(distance)
    #                 else:
    #                     distances.append(math.inf)
    #             pare_saddle_target=distances.index(min(distances))
    #         self.saddles_targets_pare_saddles_targets[i]=pare_saddle_target
    #     print(self.saddles_targets_pare_saddles_targets,"self.saddles_targets_pare_saddles_targets")

    
    def pare_saddles(self):
        """
        得到其它鞍点对于这个点的影响值的列表
         Get a list of other saddle point influence weight values for this saddle point
        """
        self.saddles_pare_saddles=[0]*self.number_of_saddles
        for saddle_facing_High in self.saddles_facing_High:
            pair_list=[]
            saddle_key=self.saddles[saddle_facing_High]
            for i in range(self.number_of_saddles):
                if i in self.saddles_facing_Low:
                    pair_list.append(self.influence_lists[saddle_key][i])
                else:
                    pair_list.append(0)
            try:
                de=1/sum(pair_list)
                pair_list=[pair*de for pair in pair_list]
            except:
                pair_list=[]
                for i in range(self.number_of_saddles):
                    if i in self.saddles_facing_Low:
                        pair_list.append(self.distances_from_saddles[saddle_key][i])
                    else:
                        pair_list.append(0)
                    min_ind=pair_list.index(min(pair_list))
                for i in range(self.number_of_saddles):
                    if i==min_ind:
                        pair_list[i]=1
                    else:
                        pair_list[i]=0
            
            self.saddles_pare_saddles[saddle_facing_High]=pair_list
            

        for saddle_facing_Low in self.saddles_facing_Low:
            pair_list=[]
            saddle_key=self.saddles[saddle_facing_Low]
            for i in range(self.number_of_saddles):
                if i in self.saddles_facing_High:
                    pair_list.append(self.influence_lists[saddle_key][i])
                else:
                    pair_list.append(0)
            try:
                de=1/sum(pair_list)
                pair_list=[pair*de for pair in pair_list]
            except:
                pair_list=[]
                for i in range(self.number_of_saddles):
                    if i in self.saddles_facing_High:
                        pair_list.append(self.distances_from_saddles[saddle_key][i])
                    else:
                        pair_list.append(0)
                    min_ind=pair_list.index(min(pair_list))
                for i in range(self.number_of_saddles):
                    if i==min_ind:
                        pair_list[i]=1
                    else:
                        pair_list[i]=0                
            
            self.saddles_pare_saddles[saddle_facing_Low]=pair_list
        print_list_with_details(self.saddles_pare_saddles,"saddles_pare_saddles")
    
    def pare_targets_with_saddles(self,NNI=False):
        if NNI:
            for i in range(self.target_high.number_of_boundaries):
                distances=[]
                saddle_items=[]
                for j,saddle in enumerate(self.saddles):

                    distance=self.target_high.get_all_distances_for_vkey(saddle)[i]
                    distances.append(distance)
                    saddle_items.append(j)
                saddle_cloest=saddle_items[distances.index(min(distances))]
                self.target_high.pare_saddles[i]=(saddle_cloest)
            print_list_with_details(self.target_high.pare_saddles,"self.target_high.pare_saddles")
            
            for i in range(self.target_low.number_of_boundaries):
                distances=[]
                saddle_items=[]
                for j,saddle in enumerate(self.saddles):

                    distance=self.target_low.get_all_distances_for_vkey(saddle)[i]
                    distances.append(distance)
                    saddle_items.append(j)
                saddle_cloest=saddle_items[distances.index(min(distances))]
                self.target_low.pare_saddles[i]=(saddle_cloest)        
            print_list_with_details(self.target_low.pare_saddles,"self.target_low.pare_saddles")
        else:
        

            for target in range(self.target_high.number_of_boundaries):
                print(target,)
                print(len(self.influence_lists[0]),)
                # print(len(self.influence_lists_High[target]))
                target_saddle_influences = [self.influence_lists[key][self.number_of_saddles+target] for key in self.saddles]
                de_sum_target_saddle_influences = 1/sum(target_saddle_influences)
                target_saddle_influences=[target_saddle_influence*de_sum_target_saddle_influences for target_saddle_influence in target_saddle_influences]              
                self.target_high.pare_saddles[target]=(target_saddle_influences)     
            print_list_with_details(self.target_high.pare_saddles,"self.target_high.pare_saddles")
            for target in range(self.target_low.number_of_boundaries):
                target_saddle_influences = [self.influence_lists[key][target+self.number_of_saddles+self.target_high.number_of_boundaries] for key in self.saddles]
                de_sum_target_saddle_influences = 1/sum(target_saddle_influences)
                target_saddle_influences=[target_saddle_influence*de_sum_target_saddle_influences for target_saddle_influence in target_saddle_influences]   
                self.target_low.pare_saddles[target]=(target_saddle_influences)  
            print_list_with_details(self.target_high.pare_saddles,"self.target_high.pare_saddles")

    def creat_saddle_offset_distances_one_side_again(self,saddles_offset_lists):
        """
        如果list facing high有none的总影响大于0.9 则保留none
        This time is to fill the offset value of saddle points who are not releverent to that set of vertices
        saddle offset list: matrix of the offset values of IS and B
        """
        for i,saddle in enumerate(self.saddles):#对于面向下的鞍点，其上侧边缘的偏移值为0，要通过周围面向上的点来补齐，反之亦然。
            if saddles_offset_lists[i] == None:
                start=False
                
                for j,saddle in enumerate(self.saddles):
                    if saddles_offset_lists[j] is not None:#遍历所有其它的点，如果不是同侧的，则根据影响力补全偏移距离表
                        if not start:#初始化
                            start=True

                            saddles_offset_list=[0]*len(saddles_offset_lists[j])
                            saddles_None_list=[0]*len(saddles_offset_lists[j])
                            for k,offset_dist in enumerate(saddles_offset_lists[j]):
                                if offset_dist is None:
                                    saddles_None_list[k]=self.saddles_pare_saddles[i][j]
                                else:
                                    saddles_offset_list[k]=offset_dist*self.saddles_pare_saddles[i][j]
                            # print_list_with_details(saddles_offset_list,"saddles_offset_list")
                            # saddles_offset_list=np.array([offset_dist*self.saddles_pare_saddles[i][j] for offset_dist in saddles_offset_lists[j] ])
                        else:
                            for k,offset_dist in enumerate(saddles_offset_lists[j]):
                                if offset_dist is None:
                                    saddles_None_list[k]+=self.saddles_pare_saddles[i][j]
                                else:
                                    #print(saddles_offset_list[k],offset_dist,self.saddles_pare_saddles[i][j],k)
                                    saddles_offset_list[k]+=offset_dist*self.saddles_pare_saddles[i][j]                            
                for k,offset_none in enumerate(saddles_None_list):
                    if offset_none>0.5:
                        saddles_offset_list[k]=None
                    else:
                        saddles_offset_list[k]/=(1-offset_none)


                saddles_offset_lists[i]=saddles_offset_list
        #print_list_with_details(saddles_offset_lists,"saddles_offset_lists")

    def creat_saddles_target_influence_field(self,method='igl'):
        """
        creat weight field matrix for all V
        """
        self.get_saddle_target_distances(method)
        self.influence_lists=self.get_saddle_target_influence_field('influences')

    def creat_saddle_influence_field(self,method='real'):
        self.get_saddle_distances(method)
        self.influence_lists=self.creat_saddle_influence_field_one_side(list(range(len(self.saddles))),'influences')
        # self.influence_lists_High=self.creat_saddle_influence_field_one_side(self.saddles_facing_High,'High_influences')
        # self.influence_lists_Low=self.creat_saddle_influence_field_one_side(self.saddles_facing_Low,'Low_influences') 
    
    def get_saddle_target_influence_field(self,attribute_name):
        """
        return a list of influence values for each saddle point
        
        """
        #influence_manual_list=[1,1,1,1,0.07,0.05,1,1,1,1]
        # influence_manual_list=[0.1,0.1,0.1,0.1,
        #                        0.1,0.1,0.1,0.1,
        #                        0.1,0.1,0.1,0.1,
        #                        0.1,0.1,0.1,0.1,
        #                        0.1,0.1,0.1,0.1,
        #                        0.1,0.1,0.05,0.05,
        #                        0.05,0.05,0.05,0.05,
        #                        0.05,0.05,0.05,0.05,]
        # influence_manual_list=[0.3,0.3,0.3,0.3,0.3,
        #                        0.3,0.3,0.3,0.3,0.3,
        #                        0.3,
        #                        1,1,1,
        #                        1,1]
        influence_manual_list=self.influence_manual_list


        
        influence_lists=[]
        saddle_target_list=list(range(self.number_of_saddles))
        min_distance_between_saddles_targets_lists=[]
        
        for saddle_target in saddle_target_list:
            distance_between_saddles_targets_list=[]
            for saddle_target2 in saddle_target_list:
                if saddle_target2 != saddle_target:
                    distance_between_saddles_targets=self.get_distance_between_saddles_targets(saddle_target,saddle_target2)
                    distance_between_saddles_targets_list.append(distance_between_saddles_targets)
            min_distance_between_saddles_targets_lists.append(min(distance_between_saddles_targets_list))

        unchange_vertex=[]
        def scale_dist(dist,scale):
            if isinstance(scale,list):
                scale0,scale1=scale
            else:
                scale0,scale1=scale,1


            if scale0==0:
                return dist
            elif dist>0:
                t=10
                dist1=((dist**t+scale0**t)**(1/t)-scale0)*scale1
                #print(dist1,dist,dist1-dist)
                return dist1
            else:
                return 0
        sum_influences=[]
        for veky in self.mesh.vertices():
            #这里是合并集合
            distance_to_all_saddles_target_veky=self.distances_from_saddles[veky]+self.distances_from_target_high[veky]+self.distances_from_target_low[veky]
            #get vi's distance to all the sources
            #print("distance_to_all_saddles_target_veky",veky,len(distance_to_all_saddles_target_veky)-len(self.distances_from_saddles[veky]),distance_to_all_saddles_target_veky[-1])
            #初始化影响数组
            # initialization of weight of vi
            influences=[0]*(self.number_of_saddles+self.target_high.number_of_boundaries+self.target_low.number_of_boundaries)
            # dist1,saddle_target1,dist2,saddle_target2=find_min_and_second_min(distance_to_all_saddles_target_veky)
            # current_distance_between_saddles_targets=self.get_distance_between_saddles_targets(saddle_target1,saddle_target2)
            # min_distance_between_saddles= min_distance_between_saddles_targets_lists[saddle_target1]
            # print_list_with_details(distance_to_all_saddles_target_veky,"distance_to_all_saddles_target_veky")


            for i,dist in enumerate(distance_to_all_saddles_target_veky):

                # actual algorithm
                try :
                    scale=influence_manual_list[i]
                except:
                    scale=1
                
                if False and self.only_normalize_big_data:
                    
                    if dist<(self.animation_frame+1)*2.185:
                        dist=dist
                    else:
                        #print(veky)
                        dist=math.inf
                elif False and self.only_normalize_big_data:
                    if dist<80:
                        dist=dist
                    else:
                        #print(veky)
                        dist=math.inf

                if isinstance(dist,float) or isinstance(dist,float):
                    dist+=0.001
                    if math.isinf(dist) or scale<-1:
                        influences[i]=0
                        #print("influence get distance inf")
                    else:
                        dist=scale_dist(dist,scale)
                        influences[i]=(1/np.sinh(self.sinha*dist))
                        #print( influences[i])
                        if self.sinha<0:
                            influences[i]=(1/(dist**(-self.sinha)))
                else:
                    print("influence get distance nan",dist)
                    dist+=0.001
                    if math.isinf(dist):
                        influences[i]=0
                        #print("influence get distance inf")
                    else:
                        dist=scale_dist(dist,scale)
                        influences[i]=(1/np.sinh(self.sinha*dist))
                        #print( influences[i])
                        if self.sinha<0:
                            influences[i]=(1/(dist**(-self.sinha)))

                # # paper
                # if i==2:
                #     influences[i]=1
                # else:
                #     influences[i]=0
                # if self.animation_frame<=200:
                #     #animation
                #     if i==2:
                #         influences[i]=self.animation_frame/100
                #     else:
                #         influences[i]=0
                # else:
                    
                #     # #animation1
                # if i==2:
                #     influences[i]=1-(self.animation_frame)/100
                # elif i==5:
                #     influences[i]=(self.animation_frame)/100
                # else:
                #     influences[i]=0






                # if i == saddle_target1:


                #     influence_unchange=1.2*((find_vetical_distance_triangle(current_distance_between_saddles_targets,dist2,dist1))/min_distance_between_saddles)
                #     influence1=influence_unchange
                #     if influence_unchange>1:
                #         unchange_vertex.append(veky)
                #         influence1=1
                #     if influence_unchange<0:
                #         influence1=0

                #     if i==self.number_of_saddles and np.isnan(influence1):
                #         print("triangle",current_distance_between_saddles_targets,dist2,dist1)


                #     influences[i]=influence1

            if self.only_normalize_big_data:
                # norlmalize for animation 0.0010342863796076358:
                sum_influence=sum(influences)
                sum_influences.append(sum_influence)
                # if sum_influence==0:
                #     print(veky)
                if sum_influence != 0:
                    de_sum=1/sum_influence             
                else:
                    de_sum=0
                influences=[x * de_sum for x in influences]  
            else:                          
                #norlmalize
                sum_influence=sum(influences) 
                sum_influences.append(sum_influence)     
                de_sum=1/sum_influence
                influences=[x * de_sum for x in influences]
            
 
                
            #  # cloest neighbor influence NNI
            # max_ind=influences.index(max(influences))
            # for indi,inf in enumerate(influences):
            #     if indi==max_ind:
            #         influences[indi]=1
            #     else:
            #         influences[indi]=0
    

            influence_lists.append(influences)
            self.mesh.vertex[veky][attribute_name]=influences
        if self.only_normalize_big_data:
            print("min_sum_inf",min(sum_influences),"max_sum_inf",max(sum_influences))
        # save_nested_list(file_name=attribute_name+'_not_smooth'+'.json',file_path=self.Outputpath,nested_list=influence_lists)
        # mesh_smooth_centroid_vertices_influence(mesh=self.mesh,fixed=unchange_vertex,kmax=100,influence_list=influence_lists)
        # save_nested_list(file_name=attribute_name+'_un_sin'+'.json',file_path=self.Outputpath,nested_list=influence_lists)
        # mici(influence_lists,2)
        #sinlize(influence_lists)
        save_nested_list(file_name=attribute_name+str(self.animation_frame)+'.json',file_path=self.Outputpath,nested_list=influence_lists)
        print('influence_size',len(influence_lists),len(influence_lists[0]))
        print_list_with_details(influence_lists,"influences lists")
        print("________________________________________________________________________________")
        return(influence_lists)            

    def get_distance_between_saddles_targets(self,saddle_target1,saddle_target2):
        """
        saddle_target_is_item_not_key
        """
        if saddle_target1==saddle_target2:
            current_distance_between_saddles_targets=0
        elif saddle_target1<self.number_of_saddles :# 1 is saddle
            saddle1_key=self.saddles[saddle_target1]
            if saddle_target2<self.number_of_saddles:# 2 is saddle
                current_distance_between_saddles_targets=self.distances_from_saddles[saddle1_key][saddle_target2]
            elif saddle_target2<self.number_of_saddles+self.target_high.number_of_boundaries: #2 is target High
                current_distance_between_saddles_targets=self.get_distance_between_one_saddle_and_target(saddle_target1,self.target_high,saddle_target2-self.number_of_saddles)
            else:#2 is target Low
                current_distance_between_saddles_targets=self.get_distance_between_one_saddle_and_target(saddle_target1,self.target_low,saddle_target2-self.number_of_saddles-self.target_high.number_of_boundaries)
        elif saddle_target1<self.number_of_saddles+self.target_high.number_of_boundaries:# 1 is target High
            if saddle_target2<self.number_of_saddles:# 2 is saddle
                current_distance_between_saddles_targets=self.get_distance_between_one_saddle_and_target(saddle_target2,self.target_high,saddle_target1-self.number_of_saddles)
            elif saddle_target2<self.number_of_saddles+self.target_high.number_of_boundaries:#2 is target HIgh 
                current_distance_between_saddles_targets=math.inf
            else:#2 is target Low
                current_distance_between_saddles_targets=self.get_distance_between_one_target_and_one_target(saddle_target1-self.number_of_saddles,saddle_target2-self.number_of_saddles-self.target_high.number_of_boundaries)
        else:# 1 is target Low
            if saddle_target2<self.number_of_saddles:# 2 is saddle
                current_distance_between_saddles_targets=self.get_distance_between_one_saddle_and_target(saddle_target2,self.target_low,saddle_target1-self.number_of_saddles-self.target_high.number_of_boundaries)
            elif saddle_target2<self.number_of_saddles+self.target_high.number_of_boundaries:#2 is target HIgh
                current_distance_between_saddles_targets=self.get_distance_between_one_target_and_one_target(saddle_target2-self.number_of_saddles,saddle_target1-self.number_of_saddles-self.target_high.number_of_boundaries)  
            else:#2 is target Low
                current_distance_between_saddles_targets=math.inf
        return current_distance_between_saddles_targets  

    def get_distance_between_one_saddle_and_target(self,saddle,target:CompoundTargetDart,boundary) :
        """
        saddle target is item not key
        """
        distances=[]
        target_veky_list = target.clustered_vkeys[boundary]
        for target_veky in target_veky_list:
            distance=self.distances_from_saddles[target_veky][saddle]
            distances.append(distance)
        current_distance_between_saddles_targets=min(distances)
        return current_distance_between_saddles_targets
    
    def get_distance_between_one_target_and_one_target(self,boundary_high,boundary_low):
        distances=[]
        target_veky_list = self.target_high.clustered_vkeys[boundary_high]
        for target_veky in target_veky_list:
            distance=self.distances_from_target_low[target_veky][boundary_low]
            distances.append(distance)
        return min(distances)
         



    def creat_saddle_influence_field_one_side(self,saddle_items,attribute_name,halfdist=200):
        print("creat_saddle_influence_field_one_side")
        influence_lists=[]
        #close_influences=[]
        for veky in self.mesh.vertices():
            all_saddles=self.distances_from_saddles[veky]
            saddles_needed=[all_saddles[item] for item in saddle_items]
            saddle_keys_needed=[self.saddles[item] for item in saddle_items]
            #close_influence=(scipy.stats.gmean(saddles_needed))
            #close_influences.append(close_influence)
        #close_influence_max=max(close_influences)
        #close_influences=[1-close_influence/close_influence_max for close_influence in close_influences]
        # print(saddle_keys_needed)
        min_distance_between_saddles_list=[]
        for i,saddle in enumerate(saddle_keys_needed):
            # print(saddle_items,i)
            saddle_items_other=copy.deepcopy(saddle_items)
            saddle_items_other.pop(i)
            # print(saddle_items_other)
            distances=[self.distances_from_saddles[saddle][saddle2] for saddle2 in saddle_items_other ]
            min_distance_between_saddles=min(distances)
            min_distance_between_saddles_list.append(min_distance_between_saddles)
        for veky in self.mesh.vertices():
            all_saddles=self.distances_from_saddles[veky]
            saddles_needed=[all_saddles[item] for item in saddle_items]
            #influences=[1 / (x + 0.001)**2 for x in saddles_needed]
            influences=[0]*(len(saddles_needed))
            #saddle1_dist,saddle1,saddle2_dist,saddle2=find_min_and_second_min(saddles_needed)
            # saddle1_key=saddle_keys_needed[saddle1]
            # current_distance_between_saddles=self.distances_from_saddles[saddle1_key][saddle2]
            # min_distance_between_saddles= min_distance_between_saddles_list[saddle1]
            for i,dist in enumerate(saddles_needed):
                if np.isinf(dist):
                    print(dist)
                    influence=0
                else:
                    influence=1/np.cosh(dist)
                
                # if i == saddle1:
                #     #influence1=1/(dist+1)

                #     print(saddle2_dist,"---",saddle1_dist)
                #     if i==self.number_of_saddles:
                #         print("triangle",current_distance_between_saddles,saddle2_dist,saddle1_dist)
                #     influence1=2*((find_vetical_distance_triangle(current_distance_between_saddles,saddle2_dist,saddle1_dist))/min_distance_between_saddles)
                #     # influence1=1
                #     #print(influence1)
                #     if influence1>1:
                #         influence1=1
                #     elif influence1<0:
                #         influence1=0
                #     # influence1=influence1**0.8
                    
                #     influence1=sinlize(influence1)




                    # influence2=max((2*halfdist-saddle1_dist)/(2*halfdist),0)
                    # influence2=sinlize(influence2)
                influences[i]=influence
                # elif i == saddle2:
                #     influences[i]=(saddle1_dist)/(saddle2_dist+saddle1_dist)*0.1
                # if i == saddle2:
                #     influence2=max((2*halfdist-saddle2_dist)/(2*halfdist),0)
                #     influence2=sinlize(influence2)
                #     influences[i]=influence2            
            sum_influence=sum(influences)      
            de_sum=1/sum_influence
            influences=[x * de_sum for x in influences]
            influence_lists.append(influences)
            self.mesh.vertex[veky][attribute_name]=influences
        save_nested_list(file_name=attribute_name+'.json',file_path=self.Outputpath,nested_list=influence_lists)
        return(influence_lists)
    def get_saddle_distances(self,method='in_mesh'):
        try:
            1/0
            
            self.distances_from_saddles=load_nested_list(file_name=(method+"distances_saddle.json"),file_path=self.Outputpath)
            print("old_distances_saddles_load_nested_list,size:",len(self.distances_from_saddles),len(self.distances_from_saddles[0]))
        except:
            for i,saddle in enumerate(self.saddles):
                if method=='in_mesh':
                    distances_from_saddle_flipped=cube_distances(mesh_input=self.mesh,mesh_close=self.close_mesh,source=saddle)
                elif method=='igl':
                    distances_from_saddle_flipped=get_igl_EXACT_geodesic_distances(mesh=self.mesh,vertices_start=[saddle])
                else:
                    distances_from_saddle_flipped=get_real_distances(mesh=self.mesh,source=saddle)

                
                self.distances_from_saddles_flipped.append(distances_from_saddle_flipped)
                # print(self.distances_from_saddles_flipped,"distances_from_saddles_flipped")

            for i in range(self.VN):
                current_values = [self.distances_from_saddles_flipped[list_index][i] for list_index in range(self.number_of_saddles)]
                self.distances_from_saddles.append(current_values) 
            print('new_saddles_distance,size:',len(self.distances_from_saddles),len(self.distances_from_saddles[0]))
            save_nested_list(file_name=(method+"distances_saddle.json"),file_path=self.Outputpath,nested_list=self.distances_from_saddles)

    def get_saddle_target_distances(self,method='in_mesh'):
        self.get_saddle_distances(method)
        try:
            1/0
            
            self.distances_from_target_high=load_nested_list(file_name=(method+"distances_target_high_voxel.json"),file_path=self.Outputpath)
            print("old distances_target_high_voxel,size:",len(self.distances_from_target_high),len(self.distances_from_target_high[0]))
        except:
            

            for boundary in range(self.target_high.number_of_boundaries):
                if method=='in_mesh':
                    distances_from_target_high_flipped=cube_distances_multi_sources(mesh_input=self.mesh,mesh_close=self.close_mesh,sources=self.target_high.clustered_vkeys[boundary])
                elif method=='igl':
                    self.distances_from_target_high_flipped=self.target_high._distances_lists
                    break                   
                else:
                    distances_from_target_high_flipped=get_real_distances(mesh=self.mesh,source=self.target_high.clustered_vkeys[boundary])
                self.distances_from_target_high_flipped.append(distances_from_target_high_flipped)
            
            for i in range(self.VN):
                current_values = [self.distances_from_target_high_flipped[list_index][i] for list_index in range(self.target_high.number_of_boundaries)]
                self.distances_from_target_high.append(current_values) 
            print("new distances_target_high_voxel,size:",len(self.distances_from_target_high),len(self.distances_from_target_high[0]))
            save_nested_list(file_name=(method+"distances_target_high_voxel.json"),file_path=self.Outputpath,nested_list=self.distances_from_target_high)           


        try:
            1/0
            print("old distances_target_low_voxel",len(self.distances_from_target_low),len(self.distances_from_target_low[0]))
            self.distances_from_target_low=load_nested_list(file_name=(method+"distances_target_low_voxel.json"),file_path=self.Outputpath)
        except:
            
            for boundary in range(self.target_low.number_of_boundaries):
                if method=='in_mesh':
                    distances_from_target_low_flipped=cube_distances_multi_sources(mesh_input=self.mesh,mesh_close=self.close_mesh,sources=self.target_low.clustered_vkeys[boundary])
                elif method=='igl':
                    self.distances_from_target_low_flipped=self.target_low._distances_lists  
                    break                
                else:
                    distances_from_target_low_flipped=get_real_distances(mesh=self.mesh,source=self.target_low.clustered_vkeys[boundary])
                self.distances_from_target_low_flipped.append(distances_from_target_low_flipped)
            
            for i in range(self.VN):

                current_values = [self.distances_from_target_low_flipped[list_index][i] for list_index in range(self.target_low.number_of_boundaries)]
                self.distances_from_target_low.append(current_values) 
            print("new distances_target_low_voxel,size",len(self.distances_from_target_low),len(self.distances_from_target_low[0]))
            save_nested_list(file_name=(method+"distances_target_low_voxel.json"),file_path=self.Outputpath,nested_list=self.distances_from_target_low)    
        self.distances_from_saddles_and_target=[self.distances_from_saddles[vk]+self.distances_from_target_high[vk]+self.distances_from_target_low[vk] for vk in range(self.VN)]
        print('distances_from_sources,size:',len(self.distances_from_saddles_and_target),len(self.distances_from_saddles_and_target[0]))
def sinlize(x,x_max=1,x_min=0):
    if isinstance(x,float ) or isinstance(x,int ):
        x_lenth=x_max-x_min
        fi=math.pi/x_lenth
        return (0.5*math.sin(fi*(x-x_min)-0.5*math.pi)+0.5)
    else:
        for i,xi in enumerate(x):
            x[i]=sinlize(xi)

        return x
def mici(x,mi):
    if isinstance(x,float ) or isinstance(x,int ):
        return (x**mi)
    else:
        for i,xi in enumerate(x):
            x[i]=mici(xi,mi)
        return x
def find_min_and_second_min(lst):  
    # 确保列表至少有两个元素  
    if len(lst) < 2:  
        raise ValueError("List must have at least two elements.")  
  
    # 使用enumerate获取元素和它们的索引  
    # 初始化最小值和倒数第二小值（假设列表非空）  
    min_val, min_idx = min((val, idx) for idx, val in enumerate(lst))  
    second_min_val = float('inf')  # 初始化为正无穷大  
    second_min_idx = None  # 初始化索引  
  
    # 遍历列表，更新最小值和倒数第二小值（如果存在）  
    for idx, val in enumerate(lst):  
        if val < second_min_val and val != min_val:  # 只考虑比当前倒数第二小值小且不等于最小值的元素  
            second_min_val, second_min_idx = val, idx  
  
    return min_val,min_idx,  second_min_val,second_min_idx
def add_on_item_to_nested_list(nested_list,item_list,number=0):
    result=[]
    for list in nested_list:
        for i,list_number  in enumerate(list):
            if i in item_list:
                list[i]+=number
        result.append(list)
    return result
def add_to_nested_list(nested_list,number=0):  
    result = []  
    for item in nested_list:  
        if isinstance(item, list):  
            # 如果元素是列表，递归调用  
            result.append(add_to_nested_list(item))  
        else:  
            # 如果元素不是列表，尝试将其与300相加  
            try:  
                result.append(item + number)  
            except TypeError:  
                # 如果元素不是数字，则保留原样或进行其他处理  
                print(f"Warning: Cannot add 300 to {item}. Skipping...")  
                result.append(item)  
    return result 
def find_vetical_distance_triangle(a,b,c):
    if b+c<a:
        return (b-c)
    p=(a+b+c)/2
    square=(p*(p-a)*(p-b)*(p-c))**0.5
    if isinstance(square,complex):
        return(b-c)
    
    h=2*square/a
    vb=(b**2-h**2)**0.5
    vc=(c**2-h**2)**0.5
    if isinstance(vb,complex):
        return(b-c)
    # print(square,vb,vc)
    return vb-a*0.5




    




    
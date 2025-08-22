from compas.datastructures import Mesh
from compas.geometry import centroid_points
#from gradient_evaluation_dart import get_vertex_gradient_from_z
#from mesh_cutting import print_list_with_details
from distances import save_nested_list
from sympy import symbols, Eq, solve
import copy
import numpy as np
import math
import time
import os
import compas_slicer.utilities as utils
from compas.files import OBJWriter 
from compas_slicer.pre_processing import create_mesh_boundary_attributes
from compas.geometry import Line, intersection_line_triangle, intersection_ray_mesh
from gradient_descent import kill_local_critical,kill_local_criticals
from collections import deque

class mesh_refiner:
    def __init__(self,mesh:Mesh,angle=math.pi/4,path=None,only_bigger=True):
        self.mesh =  mesh
        self.angle=angle
        self.path=path
        self.only_bigger=only_bigger
        
    def change_mesh_shape(self):
        if self.angle < 0:
            return False
        start=time.time()
        edited_vkey=set()
        edited_vkey.update(self.change_mesh_shape_on_one_side())
        flip_mesh(self.mesh,self.only_bigger)
        edited_vkey.update(self.change_mesh_shape_on_one_side())
        flip_mesh(self.mesh,self.only_bigger)
        print("time taken to change shape: ",time.time()-start)
        if self.path!=None:
            obj_writer = OBJWriter(filepath= os.path.join(self.path, "edited_mesh.obj"), meshes=[self.mesh])
            obj_writer.write()
        return edited_vkey

    def change_mesh_shape_on_one_side(self):
        mesh=self.mesh

        escape=0
        
        print ("start shape changing")
        check_vertices_near_boundary(mesh)
        mindist=0.2
        veky_list=list(mesh.vertices())
        all_edited_veky=set()
        for i in range(100):
            old_mesh=mesh.copy()
            self.old_mesh=old_mesh
            if i % 30==0 and i!=0:
                veky_list=list(mesh.vertices())
                mesh_smooth_centroid_vertices_z(mesh=mesh,unfixed=(edited_veky),kmax=1)
                if self.path is not None:
                    save_nested_list(self.path,'sorted_key'+str(i)+'.json',sort_veky_list)
                    save_nested_list(self.path,'edited_key'+str(i)+'.json',edited_veky)

            if mindist < 2:
                mindist+=0.05
            sort_veky_list=sorted(veky_list,key=lambda veky:mesh.vertex_attribute(name='z',key=veky))

            time,edited_veky= (self.search_mesh_shape_one_round(mesh,sort_veky_list,self.angle,mindist))
            all_edited_veky.update(edited_veky)
            list_neibors=get_unique_neighbors(mesh,edited_veky,2)
            self.check_inverse_neighbors(mesh,old_mesh,edited_veky,self.path,i)
            veky_list=list(get_all_neighbors(mesh=mesh,vertex_list=(edited_veky+list(list_neibors))))
            mesh_smooth_centroid_vertices_z(mesh=mesh,unfixed=set(list_neibors),kmax=2)
            #mesh_smooth_centroid_vertices_z(mesh=mesh,unfixed=set(list_neibors),kmax=20)
            print("round",i,"vkey searching number",len(veky_list),"accumulation veky number",time)
            if time==0:
                escape+=1
                veky_list=list(mesh.vertices())
                if escape>2:
                    break
            else:
                escape=0
    
        all_edited_veky=get_all_neighbors(mesh,all_edited_veky,5)
        if self.path != None:
            obj_writer = OBJWriter(filepath= os.path.join(self.path, "edited_mesh_r.obj"), meshes=[mesh])
            obj_writer.write()

        mesh_smooth_centroid_vertices_z(mesh=mesh,unfixed=set(all_edited_veky),kmax=2,banned_xy=True)

        mesh_smooth_centroid_vertices(mesh=mesh,unfixed=set(all_edited_veky),kmax=1)
        #mesh_smooth_centroid_vertices_z(mesh=mesh,unfixed=set(all_edited_veky),kmax=200,banned_xy=True)
        return all_edited_veky
    def search_mesh_shape_one_round(self,mesh:Mesh,veky_list,angle=math.pi/4,mindist=0.1):
        time=0
        time1=0
        edited_vkey=[]


        #print("search_mesh_shape_one_round")
        for loop_id,veky in enumerate(veky_list):
            if not mesh.vertex[veky]['near_boundary']:
                veky_normal=mesh.vertex_normal(veky)
                veky_nromal_z=veky_normal[2] 

                if veky_nromal_z>0:
                    #print("search_mesh_shape_one_round_v")
                    veky_z=mesh.vertex_attribute(name='z',key=veky)

                    neibor_list=mesh.vertex_neighbors(veky,True)
                
                    # neibormin=min(neibor_list,key=lambda neibor:get_dz_xy(mesh,veky,neibor,True))
                    # neibor_tan=get_dz_xy(mesh,veky,neibormin,True)
                    neibor_edge_list=[(neibor_list[x-1],neibor_list[x]) for x in range(len(neibor_list))]
                    tuple_list=[find_intersection_gra(mesh.vertex_coordinates(veky),mesh.vertex_coordinates(neibor_edge[0]),
                                                    mesh.vertex_coordinates(neibor_edge[1])) for neibor_edge in neibor_edge_list]
                    #print(tuple_list)
                    neibor_edge_point_zs=[x[0] for x in tuple_list]
                    neibor_edge_tan=[-x[1] for x in tuple_list]
                    neibor_edge_dzs=[x[2] for x in tuple_list]
                    neibor_edge_xys=[x[3] for x in tuple_list]
                    
                    min_index = np.argmin(neibor_edge_tan)
                
                
                    neibor_edge_min = neibor_edge_list[min_index]
                    # neibor_faces=mesh.edge_faces(neibor_edge_min[0],neibor_edge_min[1])
                    # for face_i in neibor_faces:
                    #     if veky in mesh.face_vertices(face_i):
                    #         neibor_face=face_i
                    # face_normal=mesh.face_normal(neibor_face)
                    # face_normal_angle=vector_angle_with_horizontal_plane(face_normal)
                
                    neibor_z=neibor_edge_point_zs[min_index]
                    neibor_dz=neibor_edge_dzs[min_index]
                    neibor_xy=neibor_edge_xys[min_index]
                    #print(neibor_edge_tan,neibor_edge_min)
                    # A=mesh.vertex_coordinates(veky)
                    # B=mesh.vertex_coordinates(neibor_edge_min[0])
                    # C=mesh.vertex_coordinates(neibor_edge_min[1])
                    #neibormin=min(neibor_list,key=lambda neibor:mesh.vertex_attribute(name='z',key=neibor))
                    #neibor_z=mesh.vertex_attribute(name='z',key=neibormin) 
                    #print(neibor_z,veky_z)   
                    
                    if neibor_z<veky_z:
                        #ddist=find_a_prime(A,B,C,math.pi/2-angle)
                        ddist=get_z_distance_xy_z(neibor_dz,neibor_xy,angle,mindist)
                        if ddist:
                            #print(loop_id,veky,"ddist",ddist)
                            #distlist.append(ddist)
                            mesh.vertex[veky]['z']+=ddist
                            time+=1
                            edited_vkey.append(veky)
                    else:
                        #ddist=find_a_prime(A,B,C,math.pi/2-angle)
                        ddist=get_z_distance_xy_z(neibor_dz,neibor_xy,angle,mindist)
                        #print(loop_id,veky,"pit ddist",ddist)
                        #distlist.append(ddist)
                        mesh.vertex[veky]['z']+=ddist
                        time+=1
                        time1+=1 
                        edited_vkey.append(veky)    
            else:
                pass
                #print(veky) 
                        
        # print(distlist)
        #print("buttom",time1)
        return  time,edited_vkey
    def check_inverse_neighbors(self,mesh:Mesh,old_mesh:Mesh,edited_vkeys,path,i):
        list_neibors=get_unique_neighbors(mesh,edited_vkeys,5)
        vs,fs = mesh.to_vertices_and_faces()
        for vi in edited_vkeys:
            old_vi = old_mesh.vertex_coordinates(vi)
            half_vi = [(coo+coo_old)*0.5 for coo,coo_old in zip(vs[vi],old_vi)]
            vs[vi]=half_vi
        half_edited_mesh=Mesh.from_vertices_and_faces(vs,fs)
        just_start=True
        escape=0
        search_round=1
        max_loop=50
        for loop_i in range(max_loop):
            
            if not just_start:
                old_i_v=(i_v) 
                list_neibors=get_all_neighbors(mesh,i_v,2)      
            #print(self.get_inverse_vkeys(mesh,old_mesh,half_edited_mesh,list_neibors))     
            i_v=list(self.get_inverse_vkeys(mesh,old_mesh,half_edited_mesh,list_neibors))

            if not just_start:
                if set(i_v)==set(old_i_v) :
                    escape+=1
                    if escape>5:
                        search_round+=1
                        if escape>12:
                            if path is not None:
                                save_nested_list(path,'inverse_neighbors'+str(i)+'.json',i_v)
                                save_nested_list(path,'inverse_neighbors_org_normal'+str(i)+'.json',[old_mesh.vertex_normal(key) for key in i_v])

                                obj_writer = OBJWriter(filepath= os.path.join(path, "edited_mesh"+str(i)+".obj"), meshes=[mesh])
                                obj_writer.write()
                            print("check_inverse_neighbors_maybe_finish")
                            break   
                else:
                    search_round+=1
                    escape=0   
            else:
                just_start=False                  
            print("check_inverse_neighbors",loop_i,len(i_v))
            
            if len(i_v)>0:
                smooth_vertices=get_all_neighbors(mesh,i_v,1)
                mesh_smooth_centroid_vertices_z(mesh=mesh,unfixed=smooth_vertices,damping=0.75,kmax=loop_i+2)
                for vi in smooth_vertices:
                    half_edited_mesh.vertex[vi]['z']=(mesh.vertex[vi]['z']+old_mesh.vertex[vi]['z'])*0.5
                    half_edited_mesh.vertex[vi]['y']=(mesh.vertex[vi]['y']+old_mesh.vertex[vi]['y'])*0.5
                    half_edited_mesh.vertex[vi]['x']=(mesh.vertex[vi]['x']+old_mesh.vertex[vi]['x'])*0.5
            else:
                print("check_inverse_neighbors_done")
                break
            if loop_i==max_loop-1:
                if path is not None:
                    save_nested_list(path,'inverse_neighbors'+str(i)+'.json',i_v)
                    save_nested_list(path,'inverse_neighbors_org_normal'+str(i)+'.json',[old_mesh.vertex_normal(key) for key in i_v])

                    obj_writer = OBJWriter(filepath= os.path.join(path, "edited_mesh"+str(i)+".obj"), meshes=[mesh])
                    obj_writer.write()


        
    def get_inverse_vkeys(self,mesh:Mesh,old_mesh:Mesh,half_edited_mesh,list_neibors):
        for vkey in list_neibors:
            n1=old_mesh.vertex_normal(vkey)
            n2=mesh.vertex_normal(vkey)
            n3=half_edited_mesh.vertex_normal(vkey)
            
            if not is_projection_in_acute_angle(n1,n2,n3,0):
                
                yield vkey

class mesh_refiner_x(mesh_refiner):
    def __init__(self, mesh:Mesh, angle=math.pi / 4, path=None, only_bigger=True,boundary_vs=None):
        super().__init__(mesh, angle, path, only_bigger)
        if boundary_vs is None:
            boundary_vs=get_boundary_vs(self.mesh)
            
        self.mesh_cap = get_mesh_cap(mesh,boundary_vs)
    def check_inverse_neighbors(self,mesh:Mesh,old_mesh:Mesh,edited_vkeys,path,i):
        list_neibors=get_unique_neighbors(mesh,edited_vkeys,5)

        just_start=True
        escape=0
        search_round=1
        max_loop=60
        smooth_neighbor = 0
        for loop_i in range(max_loop):
            
            if not just_start:
                old_i_v=(i_v) 
                list_neibors=get_all_neighbors(mesh,i_v,3)      
            #print(self.get_inverse_vkeys(mesh,old_mesh,half_edited_mesh,list_neibors))     
            i_v=list(self.get_inverse_vkeys(mesh,old_mesh,list_neibors=list_neibors))

            if not just_start:
                if set(i_v)==set(old_i_v) :
                    escape+=1
                    if escape>5:
                        search_round+=1
                        if escape>60:
                            if path is not None:
                                save_nested_list(path,'inverse_neighbors'+str(i)+'.json',i_v)
                                save_nested_list(path,'inverse_neighbors_org_normal'+str(i)+'.json',[old_mesh.vertex_normal(key) for key in i_v])

                                obj_writer = OBJWriter(filepath= os.path.join(path, "edited_mesh"+str(i)+".obj"), meshes=[mesh])
                                obj_writer.write()
                            print("check_inverse_neighbors_maybe_finish")
                            break   
                else:
                    search_round+=1
                    escape=0   
            else:
                just_start=False                  
            print("check_inverse_neighbors",loop_i,len(i_v))
            
            if len(i_v)>0:
                if smooth_neighbor == 0:
                    smooth_vertices=i_v
                else:
                    smooth_vertices = get_all_neighbors(mesh,i_v,smooth_neighbor)
                if loop_i < 30:
                    all_smooth_distance = mesh_smooth_centroid_vertices_z(mesh=mesh,unfixed=smooth_vertices,damping=0.75,kmax=loop_i+5)
                else:
                    all_smooth_distance = mesh_smooth_centroid_vertices(mesh=mesh,unfixed=smooth_vertices,damping=0.75,kmax=loop_i+5)
                if all_smooth_distance <= 1:
                    print('expand smooth area',all_smooth_distance)
                    smooth_neighbor+=1
                else:
                    print('all_smooth_distance',all_smooth_distance)
                    smooth_neighbor=0
   
            else:
                print("check_inverse_neighbors_done")
                break
            if loop_i==max_loop-1:
                if path is not None:
                    save_nested_list(path,'inverse_neighbors'+str(i)+'.json',i_v)
                    save_nested_list(path,'inverse_neighbors_org_normal'+str(i)+'.json',[old_mesh.vertex_normal(key) for key in i_v])

                    obj_writer = OBJWriter(filepath= os.path.join(path, "edited_mesh"+str(i)+".obj"), meshes=[mesh])
                    obj_writer.write()
    
    def get_inverse_vkeys(self, mesh, old_mesh=None, half_edited_mesh=None, list_neibors=None):
        fs = mesh.to_vertices_and_faces()
        cfs = self.mesh_cap.to_vertices_and_faces()
        for vkey in list_neibors:
            n = self.mesh.vertex_normal(vkey)
            
            p = self.mesh.vertex_coordinates(vkey)
            #print(intersection_ray_mesh((p,n),fs))
            a=intersection_ray_mesh((p,n),fs)
            b=intersection_ray_mesh((p,n),cfs)
            a=[tuple[0] for tuple in a if tuple[0] not in mesh.vertex_faces(vkey)]
            inter_number = len(a)+len(b)
            if inter_number%2==1:
                #print(len(b))
                yield vkey

class mesh_refiner_xx(mesh_refiner_x):
    def __init__(self, mesh:Mesh, angle=math.pi / 4, path=None, only_bigger=True,boundary_vs=None,angle_scale=1.05,edge_angle=math.pi*0.333):
        super().__init__(mesh, angle, path, only_bigger,boundary_vs)
        self.edge_angle_tan = math.cos(edge_angle*0.5)*math.sin(angle)
        self.edge_angle_tan = self.edge_angle_tan/(1-self.edge_angle_tan)**0.5
        self.angle_scale=angle_scale*self.angle
    def change_mesh_shape(self):
        if self.angle < 0:
            return False
        start=time.time()
        edited_vkey=set()
        self.edge_angle_tan_org = self.edge_angle_tan
        self.edge_angle_tan = 0.1*self.edge_angle_tan
        edited_vkey.update(self.change_mesh_shape_on_one_side())
        flip_mesh(self.mesh,self.only_bigger)
        flip_mesh(self.mesh_cap,self.only_bigger)
        if self.only_bigger:
            self.angle = 0.1*self.angle
            self.edge_angle_tan=self.edge_angle_tan_org
        edited_vkey.update(self.change_mesh_shape_on_one_side())
        flip_mesh(self.mesh,self.only_bigger)
        print("time taken to change shape: ",time.time()-start)
        if self.path!=None:
            obj_writer = OBJWriter(filepath= os.path.join(self.path, "edited_mesh.obj"), meshes=[self.mesh])
            obj_writer.write()
        return edited_vkey    
    def search_mesh_shape_one_round(self,mesh:Mesh,veky_list,angle=math.pi/4,mindist=0.1):
        self.mindist = mindist
        time=0
     
        edited_vkey=[]



        #print("search_mesh_shape_one_round")
        for loop_id,vkey in enumerate(veky_list):
            time,current_edited_vkeys = self.search_vkey_area(vkey,time)
            if current_edited_vkeys:
                edited_vkey.extend(current_edited_vkeys)
        # print(distlist)
        #print("buttom",time1)
        return  time,edited_vkey 
    
    def search_vkey_area(self,start_vkey,time):
        all_edited_vkey=set()
        quene = deque()
        quene.append(start_vkey)

        while quene:
            #print(quene)
            vkey = quene.popleft()
            time,new_editted_vkey = self.search_vkey(vkey,time)
            #print("__",new_editted_vkey)
            if new_editted_vkey is not None:
                quene.extend(new_editted_vkey)
                all_edited_vkey.update(new_editted_vkey)
        return time,all_edited_vkey

    
    def search_vkey(self,vkey,time):
        
        if not self.mesh.vertex[vkey]['near_boundary']:
            veky_normal=self.mesh.vertex_normal(vkey)
            veky_nromal_z=veky_normal[2] 

            if veky_nromal_z>0:
                #print("search_mesh_shape_one_round_v")
                veky_z=self.mesh.vertex_attribute(name='z',key=vkey)

                neibor_list=self.mesh.vertex_neighbors(vkey,True)

                neibor_edge_list=[(neibor_list[x-1],neibor_list[x]) for x in range(len(neibor_list))]
                tuple_list=[find_intersection_gra(self.mesh.vertex_coordinates(vkey),self.mesh.vertex_coordinates(neibor_edge[0]),
                                                self.mesh.vertex_coordinates(neibor_edge[1])) for neibor_edge in neibor_edge_list]

                neibor_edge_point_zs=[x[0] for x in tuple_list]
                neibor_edge_tan=[-x[1] for x in tuple_list]
                neibor_edge_dzs=[x[2] for x in tuple_list]
                neibor_edge_xys=[x[3] for x in tuple_list]
                
                min_index = np.argmin(neibor_edge_tan)
            
            
                neibor_edge_min = neibor_edge_list[min_index]

            
                neibor_z=neibor_edge_point_zs[min_index]
                neibor_dz=neibor_edge_dzs[min_index]
                neibor_xy=neibor_edge_xys[min_index]

                
         

                ddist=get_z_distance_xy_z(neibor_dz,neibor_xy,self.angle_scale,self.mindist)
                if ddist:

                    self.mesh.vertex[vkey]['z']+=ddist
                    time+=1
                    editted_vkey=[vkey]
                    path_vkeys =self.refine_way_up(vkey)
                    if path_vkeys:
                        mesh_smooth_centroid_vertices_z(mesh=self.mesh,unfixed=get_unique_neighbors(self.mesh,path_vkeys,1),kmax=5)
                        mesh_smooth_centroid_vertices_z(mesh=self.mesh,unfixed=get_all_neighbors(self.mesh,path_vkeys,1),kmax=5)
                    editted_vkey.extend(path_vkeys)
                    #print('_',editted_vkey)
                else:
                    editted_vkey=None
            else:
                editted_vkey=None
        else:
            editted_vkey=None
            #print(veky) 
        return time,editted_vkey 
    
    def refine_way_up(self,vkey):
        if not self.check_way_up_tan(vkey):
            path = kill_local_critical(self.mesh,vkey,'zs',slope_need=self.edge_angle_tan)
        else:
            path = []
        return path

    def check_way_up_tan(self,vkey):
        for n in self.mesh.vertex_neighbors(vkey,True):
            if self.mesh.vertex[n]['z']>self.mesh.vertex[vkey]['z']:
                if get_slope(self.mesh,n,vkey) > self.edge_angle_tan:
                    return True   
        return False  
    def get_inverse_vkeys(self, mesh, old_mesh=None, half_edited_mesh=None, list_neibors=None):
        fs = mesh.to_vertices_and_faces()
        cfs = self.mesh_cap.to_vertices_and_faces()
        for vkey in list_neibors:
            n = self.mesh.vertex_normal(vkey)
            n_0 =self.old_mesh.vertex_normal(vkey)
            if np.dot(n,n_0)<0:
                p = self.mesh.vertex_coordinates(vkey)
                #print(intersection_ray_mesh((p,n),fs))
                a=intersection_ray_mesh((p,n),fs)
                b=intersection_ray_mesh((p,n),cfs)
                a=[tuple[0] for tuple in a if tuple[0] not in mesh.vertex_faces(vkey)]
                inter_number = len(a)+len(b)
                if inter_number%2==1:
                    #print(len(b))
                    yield vkey
          
def get_mesh_cap(mesh:Mesh,cluster_vs):
    from distances import rearrange_points
    cap_vertices=[]
    cap_vertices_key_in_mesh=[]
    cap_faces=[]
    if not isinstance(cluster_vs[0],list):
        cluster_vs=[cluster_vs]
    for cluster_org in cluster_vs:
        cluster_org = rearrange_points(mesh,cluster_org)
        cap_vertices.extend([mesh.vertex_coordinates(key) for key in cluster_org])
        cap_vertices_key_in_mesh.extend(cluster_org)
        cluster_edges=[edge for edge in mesh.edges() if edge[0] in cluster_org and edge[1] in cluster_org]
        coordinates=[]
        for veky in cluster_org:
            coordinates.append(np.array(mesh.vertex_coordinates(key=veky)))
        average_coordinates = list(np.mean(coordinates, axis=0))


        cap_vertices.append(average_coordinates)
        cap_vertices_key_in_mesh.append(-1)
    
        for i,edge in enumerate(cluster_edges):
 
            cap_faces.append([cap_vertices_key_in_mesh.index(edge[0]),cap_vertices_key_in_mesh.index(edge[1]),(len(cap_vertices)-1)])
    print(cap_vertices,cap_faces)
    mesh_cap = Mesh.from_vertices_and_faces(cap_vertices, cap_faces)

    return mesh_cap       
def is_projection_in_acute_angle(n1, n2, n3, tol=1e-8):
    """
    判断向量 n3 在 n1 和 n2 张成的平面上的投影是否位于 n1 和 n2 的较小夹角之间。
    即，投影是否满足 a > 0 且 b > 0（n3 的投影 = a*n1 + b*n2）。

    Parameters:
        n1, n2, n3: 输入向量（np.array 或 list）。
        tol: 数值计算容差（避免浮点误差）。

    Returns:
        bool: 如果投影位于较小夹角内（a > 0 且 b > 0），返回 True；否则返回 False。
    """
    # 转换为 NumPy 数组
    
    n1 = np.array(n1, dtype=float)
    n2 = np.array(n2, dtype=float)
    n3 = np.array(n3, dtype=float)
    if (n1==n2).all():
        return True
    # 构造线性方程组求解 a 和 b
    # 方程组：
    #   n3 · n1 = a (n1 · n1) + b (n2 · n1)
    #   n3 · n2 = a (n1 · n2) + b (n2 · n2)
    A = np.array([
        [np.dot(n1, n1), np.dot(n2, n1)],
        [np.dot(n1, n2), np.dot(n2, n2)]
    ])
    b = np.array([np.dot(n3, n1), np.dot(n3, n2)])

    # 解方程组 Ax = b
    try:
        a, b = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # 如果 n1 和 n2 线性相关（共线），则无法形成平面，直接返回 False
        #print(n1,n2,"n1 和 n2 线性相关（共线），无法形成平面")
        if np.dot(n1,n2)>0:
            return True
        else:
            return False

    # 判断 a > 0 且 b > 0（考虑浮点误差，用 tol 判断）
    return (a > tol) and (b > tol)
def flip_mesh(mesh:Mesh,only_bigger=True):
    """
    if only bigger, flip the z direction
    if not only bigger, flip the normal
    """
    if only_bigger:
        for vertex,data in mesh.vertices(data=True):
            data['z']=-data['z']
    mesh.flip_cycles()



# def search_mesh_shape_one_round(mesh:Mesh,veky_list,angle=math.pi/4):
#     time=0
#     time1=0
#     edited_vkey=[]


#     #print("search_mesh_shape_one_round")
#     for veky in veky_list:
#         if not mesh.vertex[veky]['near_boundary']:
#             #print("search_mesh_shape_one_round_v")
#             veky_z=mesh.vertex_attribute(name='z',key=veky)
#             veky_boundary=mesh.vertex_attribute(name='boundary',key=veky)
#             neibor_list=mesh.vertex_neighbors(veky)
#             neibormin=min(neibor_list,key=lambda neibor:mesh.vertex_attribute(name='z',key=neibor))
#             neibor_z=mesh.vertex_attribute(name='z',key=neibormin) 
#             veky_normal=mesh.vertex_normal(veky)
#             veky_nromal_z=veky_normal[2]      
   

#             if veky_nromal_z>0:
#                 if neibor_z<veky_z:
#                     ddist=get_z_distance(mesh,neibormin,veky,angle)
#                     if ddist:
#                         #distlist.append(ddist)
#                         mesh.vertex[veky]['z']+=ddist
#                         time+=1
#                         edited_vkey.append(veky)
#                 else:
#                     ddist=get_z_distance(mesh,neibormin,veky,angle)
#                     #distlist.append(ddist)
#                     mesh.vertex[veky]['z']+=ddist
#                     time+=1
#                     time1+=1 
#                     edited_vkey.append(veky)    
#         else:
#             print(veky) 
                    
#     # print(distlist)
#     #print("buttom",time1)
#     return  time,edited_vkey
def point_to_line_distance(A, B, C):
    A=np.array(A)
    B=np.array(B)
    C=np.array(C)
    # 计算向量
    AB = B-A
    AC = C-A
    BC = C-B
    
    # 计算参数t
    t = np.dot(AB, AC) / np.dot(AB, AB)
    
    # 计算垂足D的坐标
    D = np.array(B) + t * BC
 

    
    # 计算垂线AD与平面的夹角的正切值
    dz=(A[2] - D[2]) 
    xy= math.sqrt((A[0] - D[0])**2 +(A[1] - D[1])**2)
    tan_theta = dz / xy
    
    return D[2], tan_theta,dz,xy
def find_intersection_gra(A, B, C):

    # 将点转换为numpy数组
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    
    # 计算向量AB和AC
    AB = B - A
    AC = C - A
    
    # 计算法向量n
    n = np.cross(AB, AC)
    n_x, n_y, n_z = n

    
    # 计算t

    numerator = (n_x*(B[1]-A[1])+n_y*(A[0]-B[0]))
    denominator = -(n_x*(C[1]-B[1])+n_y*(B[0]-C[0]))
    
    if denominator == 0:

        #print("分母为0，无法计算t,取0避免bug")
        if C[2]>B[2]:
            t=0
        else:
            t=1
    else:
    
        t = numerator / denominator
    
    # 计算交点坐标
    if t <0:
        D=B
    elif t>1:
        D=C
    else:
        D = B + t * np.array(C-B)
    
    # 计算垂线AD与平面的夹角的正切值
    dz=(A[2] - D[2]) 
    xy= math.sqrt((A[0] - D[0])**2 +(A[1] - D[1])**2)
    tan_theta = dz / xy
    
    return D[2], tan_theta,dz,xy
    

def get_xy_distance(mesh:Mesh,u,v):
    u_x,u_y=mesh.vertex_attributes(names=['x','y'],key=u)
    v_x,v_y=mesh.vertex_attributes(names=['x','y'],key=v)
    return(((u_x-v_x)**2+(u_y-v_y)**2)**0.5)

def get_slope(mesh:Mesh,u,v):
    u_x,u_y,u_z=mesh.vertex_attributes(names=['x','y','z'],key=u)
    v_x,v_y,v_z=mesh.vertex_attributes(names=['x','y','z'],key=v)
    return (u_z-v_z)/(((u_x-v_x)**2+(u_y-v_y)**2)**0.5)

def get_z_distance(mesh:Mesh,u,v,angle=math.pi/4):
    u_z=mesh.vertex_attribute(name='z',key=u)
    v_z=mesh.vertex_attribute(name='z',key=v)
    d_z=v_z-u_z
    distance_xy=get_xy_distance(mesh,u,v)
    z_needed=math.tan(angle)*distance_xy
    if d_z<z_needed:
        return max(z_needed-d_z,0.1)
    else:
        return False
def get_dz_xy(mesh:Mesh,u,v,tan=False):
    u_z=mesh.vertex_attribute(name='z',key=u)
    v_z=mesh.vertex_attribute(name='z',key=v)
    d_z=v_z-u_z
    distance_xy=get_xy_distance(mesh,u,v)
    if tan:
        return d_z/distance_xy
    return d_z,distance_xy    
def get_z_distance_xy_z(d_z,distance_xy,angle=math.pi/4,mindist=0.1):

   
    z_needed=math.tan(angle)*distance_xy
    if d_z<z_needed:
        return max(z_needed-d_z,mindist)
    else:
        return False

def get_unique_neighbors(mesh:Mesh, vertex_list,round=3):
    unique_neighbors = set()

    # Step through each vertex in the list
    for vertex in vertex_list:
        #print(vertex)
        veky_boundary=mesh.vertex_attribute(key=vertex,name='boundary')
        if  (veky_boundary != 1 and veky_boundary != 2):
            # Get neighbors of the current vertex
            neighbors = mesh.vertex_neighborhood(vertex,round)
            neighbors = list(check_vertices_on_boundary(mesh,neighbors))
            # neighbors = list(check_vertices_normal(mesh,neighbors))
            # Add neighbors to the set
            unique_neighbors.update(neighbors)
    
    # Remove vertices that are already in the input list
    unique_neighbors.difference_update(vertex_list)
    
    return unique_neighbors

def get_all_neighbors(mesh:Mesh, vertex_list,n=2):
    unique_neighbors = set()

    # Step through each vertex in the list
    for vertex in vertex_list:
        veky_boundary=mesh.vertex_attribute(key=vertex,name='boundary')
        if  (veky_boundary != 1 and veky_boundary != 2):
            # Get neighbors of the current vertex
            neighbors = mesh.vertex_neighborhood(vertex,n)
            vailid_neighbors=[]
            for i in neighbors:
                if not mesh.vertex[i]['near_boundary']:
                    vailid_neighbors.append(i)

            # Add neighbors to the set
            unique_neighbors.update(vailid_neighbors)    
    return unique_neighbors    

def check_vertices_on_boundary(mesh:Mesh,vertices):
    for veky in vertices:
        veky_boundary=mesh.vertex_attribute(key=veky,name='near_boundary')
        if  not veky_boundary:
            yield(veky)   

def check_vertices_normal(mesh:Mesh,vertices) :
    for veky in vertices:
        veky_normal_z=mesh.vertex_normal(key=veky)[2]
        if  veky_normal_z<0:
            yield(veky)     

def check_vertices_near_boundary(mesh:Mesh):
    boundarys=[]
    for veky in mesh.vertices():
        mesh.vertex[veky]['near_boundary']=False
        veky_boundary=mesh.vertex_attribute(key=veky,name='boundary')
        if (veky_boundary == 1 or veky_boundary == 2):
             boundarys.append(veky) 
    print(len(boundarys))
    for veky in boundarys:
        mesh.vertex[veky]['near_boundary']=True
        for neibor in mesh.vertex_neighborhood(key=veky,ring=1):
            mesh.vertex[neibor]['near_boundary']=True              




def mesh_smooth_centroid_vertices(mesh:Mesh, unfixed, kmax=10, damping=0.5, callback=None, callback_args=None,centriod=True):
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
    all_distance=0
    if callback:
        if not callable(callback):
            raise Exception("Callback is not callable.")


    for k in range(kmax):
        key_xyz = {key: mesh.vertex_coordinates(key) for key in mesh.vertices()}

        for key in unfixed:
            if not mesh.vertex[key]['near_boundary']:
                attr=mesh.vertex_attributes(key=key)
                if key in unfixed:
                    x0, y0, z0 = key_xyz[key]

                    a,b,c,d=fit_plane([key_xyz[nbr] for nbr in mesh.vertex_neighbors(key)])
                    if centriod:
                        xm,ym,zm=centroid_points([key_xyz[nbr] for nbr in mesh.vertex_neighbors(key)])
                    else:
                        xm,ym,zm=x0, y0, z0

                    cx, cy, cz = project_point_to_plane(a,b,c,d,xm,ym,zm)
                    dx = cx - x0
                    dy = cy - y0
                    dz = cz - z0

                    attr["x"] += damping * dx
                    attr["y"] += damping * dy
                    attr["z"] += damping * dz
                    all_distance+=(dx**2+dy**2+dz**2)**0.5
        if callback:
            callback(k, callback_args)
    return all_distance
def mesh_smooth_centroid_vertices_z(mesh:Mesh, unfixed, kmax=10, damping=0.5, callback=None, callback_args=None,banned_xy=False):
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
    all_distance=0
    if callback:
        if not callable(callback):
            raise Exception("Callback is not callable.")


    for k in range(kmax):
        key_xyz = {key: mesh.vertex_coordinates(key) for key in mesh.vertices()}

        for key in unfixed:
            attr=mesh.vertex_attributes(key=key)
            if key in unfixed:
                if not mesh.vertex[key]['near_boundary']:
                    x0, y0, z0 = key_xyz[key]
               
                    a,b,c,d=fit_plane([key_xyz[nbr] for nbr in mesh.vertex_neighbors(key)])
                  
                    xm,ym,zm=centroid_points([key_xyz[nbr] for nbr in mesh.vertex_neighbors(key)])
                

                    cx, cy, cz = project_point_to_plane(a,b,c,d,xm,ym,zm)
                    if not banned_xy:
                        dx=damping * (cx - x0)
                        dy=damping * (cy - y0)
                        attr["x"] += dx
                        attr["y"] += dy
                    else:
                        dx=0
                        dy=0
                    if cz>z0:
                        dz=damping * (cz - z0)
                        attr["z"] += dz
                    else:
                        dz=0
                    all_distance+=(dx**2+dy**2+dz**2)**0.5

        if callback:
            callback(k, callback_args)
    return all_distance

def fit_plane(points):
    """
    用最小二乘法拟合空间平面，支持竖直平面
    :param points: 嵌套列表，每个元素是 [x, y, z]
    :return: 平面方程系数 a, b, c, d (ax + by + cz + d = 0)
    """
    # 将数据转换为 numpy 数组
    points = np.array(points)
    x = points[:, 0]  # 所有点的 x 坐标
    y = points[:, 1]  # 所有点的 y 坐标
    z = points[:, 2]  # 所有点的 z 坐标

    # 构建矩阵 A
    A = np.vstack([x, y, z, np.ones(len(x))]).T

    # 对 A 进行奇异值分解 (SVD)
    _, _, Vt = np.linalg.svd(A)

    # 最小奇异值对应的右奇异向量即为平面方程系数
    a, b, c, d = Vt[-1]

    # 归一化系数
    norm = np.sqrt(a**2 + b**2 + c**2)
    a, b, c, d = a / norm, b / norm, c / norm, d / norm

    return a, b, c, d
def project_point_to_plane(a, b, c, d, x0, y0, z0):
    # 计算分母，避免重复计算
    denominator = a**2 + b**2 + c**2
    
    # 确保分母不为0，避免除以0的情况
    if denominator == 0:
        raise ValueError("Plane normal cannot be zero vector.")
    
    # 计算分子
    numerator = a*x0 + b*y0 + c*z0 + d
    
    # 计算垂足点坐标
    xp = x0 - a * (numerator / denominator)
    yp = y0 - b * (numerator / denominator)
    zp = z0 - c * (numerator / denominator)
    
    return (xp, yp, zp)

def find_a_prime(A, B, C, theta):
    # 计算BC的方向向量
    BC = np.array(C) - np.array(B)
    
    # 水平面的单位法向量
    n_horizontal = np.array([0, 0, 1])
    
    # 计算垂直于BC和平面法线的向量
    n_perpendicular = np.cross(BC, n_horizontal)
    
    # 标准化这个向量
    n_perpendicular_normalized = n_perpendicular /math.sqrt((n_perpendicular[0])**2 +(n_perpendicular[1])**2)
    
    # 根据yt计算平面法线向量的z分量
    sin_theta = np.sin(theta)
    cos_theta =np.cos(theta)
    
    # 调整n_perpendicular_normalized以满足sin(theta)的要求
    n_adjusted_positive = n_perpendicular_normalized * (cos_theta / abs(n_perpendicular_normalized[2])) if n_perpendicular_normalized[2] != 0 else n_perpendicular_normalized
    n_adjusted_positive[2] = sin_theta * np.sign(n_perpendicular_normalized[2])
    
    n_adjusted_negative = n_perpendicular_normalized * (cos_theta / abs(n_perpendicular_normalized[2])) if n_perpendicular_normalized[2] != 0 else n_perpendicular_normalized
    n_adjusted_negative[2] = -sin_theta * np.sign(n_perpendicular_normalized[2])
    
    # 构造两个可能的平面方程
    def plane_equation(n, point):
        return lambda x, y, z: n[0]*(x - point[0]) + n[1]*(y - point[1]) + n[2]*(z - point[2])
    
    plane1 = plane_equation(n_adjusted_positive, B)
    plane2 = plane_equation(n_adjusted_negative, B)
    
    # A点坐标
    x_a, y_a, z_a = A
    
    # 对于每个平面，求解A点在该平面上的投影点A'
    def solve_for_y(plane, x, z):
        from sympy import symbols, Eq, solve
        y = symbols('y')
        eq = Eq(plane(x, y, z), 0)
        sol = solve(eq, y)
        return sol[0] if sol else None
    
    y1 = solve_for_y(plane1, x_a, z_a)
    y2 = solve_for_y(plane2, x_a, z_a)
    
    # 计算A到A'的距离
    dist1 = (y_a - y1) if y1 is not None else float('inf')
    dist2 = (y_a - y2) if y2 is not None else float('inf')
    if dist1 <0 and dist2<0:
        return False
    else:

        return max(dist1, dist2,0.1)
    #a_prime = [x_a, y1, z_a] if dist1 < dist2 else [x_a, y2, z_a]

def vector_angle_with_horizontal_plane(v):
    # 确保输入是一个numpy数组
    v = np.array(v)
    
    # 计算向量的模长
    v_magnitude = np.linalg.norm(v)
    
    # 计算向量与水平面法线向量 [0, 0, 1] 的点积
    dot_product = v[2]
    
    # 计算夹角的余弦值
    cos_theta = dot_product / v_magnitude
    
    # 计算夹角（以弧度为单位）
    theta = np.arccos(cos_theta)
    
    return theta

def get_boundary_vs(mesh:Mesh):
    naked=set()
    for edge in mesh.edges():
        #print(edge)
        #print(mesh.edge_faces(edge[0],edge[1]))
        faces=mesh.edge_faces(edge[0],edge[1])
        if None in faces:
            naked.update(edge)
            #print(edge,mesh.edge_faces(edge[0],edge[1]))
    from Target_finding import group_connected_elements
    cluster_vs=group_connected_elements(naked,list(mesh.edges()))
    return cluster_vs

def main():

    input_folder_name='beam_testprint'#'beam_new''min1''beam1B''whole''example_jun_bg''data_Y_shape' 'data_vase''data_costa_surface''data_Y_shape_o''data_vase_o''data_costa_surface_o''Jun_ab_testmultipipe'
    #'Jun_ah_testb''Jul_ai''Jul_h''Jul_I''Jul_ab''Jul_ah''Jul_ba''table_1''table_2''Aug_ac_ex''Aug_bg''Aug_bh''example_jun_bg'
    DATA_PATH = os.path.join(os.path.dirname(__file__), input_folder_name)
    OUTPUT_PATH = utils.get_output_directory(DATA_PATH)
    OBJ_INPUT_NAME = os.path.join(DATA_PATH, 'mesh.obj')
        
    # --- Load initial_mesh
    mesh = Mesh.from_obj(os.path.join(DATA_PATH, OBJ_INPUT_NAME))
    # --- Load targets (boundaries)
    try:
        1/0
        low_boundary_vs = utils.load_from_json(DATA_PATH, 'boundaryLOW.json')
        high_boundary_vs = utils.load_from_json(DATA_PATH, 'boundaryHIGH.json')
    except:
        try:
            1/0
            high_boundary_vs = utils.load_from_json(DATA_PATH, 'boundaryHIGH.json')
            low_boundary_vs = []
        except:

            naked=set()
            for edge in mesh.edges():
                #print(edge)
                #print(mesh.edge_faces(edge[0],edge[1]))
                faces=mesh.edge_faces(edge[0],edge[1])
                if None in faces:
                    naked.update(edge)
                    #print(edge,mesh.edge_faces(edge[0],edge[1]))
                # print(mesh.edge_faces(edge[0],edge[1]))
                # if mesh.edge_faces(edge[0],edge[1])[0]==mesh.edge_faces(edge[0],edge[1])[1]:
                #     1/0
                
                #naked.update(edge)
                

            low_boundary_vs = list(naked)
            high_boundary_vs=[]
            print(low_boundary_vs)

    create_mesh_boundary_attributes(mesh, low_boundary_vs, high_boundary_vs)  
    from Target_finding import group_connected_elements
    cluster_vs = group_connected_elements(low_boundary_vs, list(mesh.edges()))

    for vi in mesh.vertices():
        mesh.vertex[vi]['y']*=1
    #from gradient_descent import kill_local_criticals
    mesh_modifier = mesh_refiner_xx(mesh,math.pi*0.01,OUTPUT_PATH,only_bigger=True,boundary_vs=cluster_vs,angle_scale=1.1)
    #mesh_modifier = mesh_refiner(mesh,math.pi/3,OUTPUT_PATH,only_bigger=False)
    # flip_mesh(mesh,True)
    # edited_vkeys=change_mesh_shape(mesh,math.pi/4,OUTPUT_PATH,only_bigger=True)
    # flip_mesh(mesh,True)
    # edited_vkeys=change_mesh_shape(mesh,math.pi/4,OUTPUT_PATH,only_bigger=False)
    # flip_mesh(mesh,True)
    # edited_vkeys=change_mesh_shape(mesh,math.pi/3,OUTPUT_PATH,only_bigger=True)
    # flip_mesh(mesh,True)
    # edited_vkeys=change_mesh_shape(mesh,math.pi/3,OUTPUT_PATH,only_bigger=False)
    # flip_mesh(mesh,True)
    # edited_vkeys=change_mesh_shape(mesh,math.pi/2.8,OUTPUT_PATH,only_bigger=True)
    # flip_mesh(mesh,True)
    edited_vkeys=mesh_modifier.change_mesh_shape()
    from gradient_evaluation_dart import GradientEvaluation_Dart
    G=GradientEvaluation_Dart(mesh,DATA_PATH)
    for vi in mesh.vertices():
        mesh.vertex[vi]['scalar_field']=mesh.vertex[vi]['z']
    G.find_critical_points()

    kill_local_criticals(mesh,G.maxima,[],'z')


    # mesh_modifier = mesh_refiner_xx(mesh,math.pi*0.31,OUTPUT_PATH,only_bigger=True,boundary_vs=cluster_vs,angle_scale=1.1)
    # edited_vkeys=mesh_modifier.change_mesh_shape()
    # for vi in mesh.vertices():
    #     mesh.vertex[vi]['scalar_field']=mesh.vertex[vi]['z']
    # G.find_critical_points()

    # kill_local_criticals(mesh,G.maxima,[],'z')

    # mesh_modifier = mesh_refiner_xx(mesh,math.pi*0.32,OUTPUT_PATH,only_bigger=True,boundary_vs=cluster_vs,angle_scale=1.1)
    # edited_vkeys=mesh_modifier.change_mesh_shape()
    # for vi in mesh.vertices():
    #     mesh.vertex[vi]['scalar_field']=mesh.vertex[vi]['z']
    # G.find_critical_points()

    # kill_local_criticals(mesh,G.maxima,[],'z')

    # mesh_modifier = mesh_refiner_xx(mesh,math.pi*0.33,OUTPUT_PATH,only_bigger=False,boundary_vs=cluster_vs,angle_scale=1.1)
    # edited_vkeys=mesh_modifier.change_mesh_shape()
    # for vi in mesh.vertices():
    #     mesh.vertex[vi]['scalar_field']=mesh.vertex[vi]['z']
    # G.find_critical_points()

    # kill_local_criticals(mesh,G.maxima,[],'z')

    for vi in mesh.vertices():
        mesh.vertex[vi]['y']*=1
    #mesh_smooth_centroid_vertices(mesh=mesh,unfixed=set(edited_vkeys),kmax=10,centriod=False)
    
    if OUTPUT_PATH!=None:
        obj_writer = OBJWriter(filepath= os.path.join(OUTPUT_PATH, "edited_mesh.obj"), meshes=[mesh])
        obj_writer.write()

    # print(mesh.vertex_attribute(key=6615, name="z"))


if __name__ == "__main__":
    main()
 
def centroid_points(points):
    """Compute the centroid of a set of points.

    Parameters
    ----------
    points : sequence[[float, float, float] | :class:`~compas.geometry.Point`]
        A sequence of XYZ coordinates.

    Returns
    -------
    [float, float, float]
        XYZ coordinates of the centroid.

    Warnings
    --------
    Duplicate points are **NOT** removed. If there are duplicates in the
    sequence, they should be there intentionally.

    Examples
    --------
    >>> points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    >>> centroid_points(points)
    [0.5, 0.5, 0.0]
    """
    p = len(points)
    x, y, z = zip(*points)
    return [sum(x) / p, sum(y) / p, sum(z) / p]

"""

# 参数
R = 10  # 网格的半径
theta = np.radians(30)  # 安息角

# 初始化地形高度
grid_size = 100
heights = np.zeros((grid_size, grid_size))

# 计算法向量和倾角的函数
def calculate_slope(heights, i, j):
    # 假设网格是规则的，计算相邻点的高度差
    dh_dx = (heights[i+1, j] - heights[i-1, j]) / 2
    dh_dy = (heights[i, j+1] - heights[i, j-1]) / 2
    slope = np.sqrt(dh_dx**2 + dh_dy**2)
    return slope

# 迭代更新高度
def update_heights(heights, max_iterations=1000, delta_h=0.1):
    for _ in range(max_iterations):
        changes = 0
        for i in range(1, grid_size-1):
            for j in range(1, grid_size-1):
                slope = calculate_slope(heights, i, j)
                if slope > np.tan(theta):
                    heights[i, j] += delta_h
                    changes += 1
        if changes == 0:
            break
    return heights

# 执行高度更新
heights = update_heights(heights)

# 可视化结果
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-R, R, grid_size)
y = np.linspace(-R, R, grid_size)
x, y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, heights, cmap='terrain')
plt.show()

"""
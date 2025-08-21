from compas.datastructures import Mesh
import compound_target_dart as Compound_Target
from typing import List
from collections import deque


class related_boundary_finder:
    def __init__(self,mesh:Mesh,vertex,target_high:Compound_Target,target_low:Compound_Target):
        self.mesh=mesh
        self.target_high=target_high
        self.target_low=target_low

    def find_saddle_point_with_related_boundary(self):
        pass
def gradient_descent(mesh:Mesh,vertex,name='scalar_field',direction=True,drump_out=True):
    
    #print('gradient_descent',vertex)
    neighbors = mesh.vertex_neighbors(vertex)
    # 使用列表推导式找到 name 值最小的 neighbor
    if direction:
        next_vertex = max(neighbors, key=lambda neighbor: mesh.vertex[neighbor][name])
        if mesh.vertex_attribute(key=next_vertex,name=name) < mesh.vertex_attribute(key=vertex,name=name):
            if drump_out:
                next_vertex=drump_out_local_critical(mesh=mesh,current_ring={next_vertex},name=name,slope_need=0,direction=direction)
            else:
                raise ValueError('vertex is local maximun')
     
    else:
        next_vertex = min(neighbors, key=lambda neighbor: mesh.vertex[neighbor][name])
        if mesh.vertex_attribute(key=next_vertex,name=name) > mesh.vertex_attribute(key=vertex,name=name):
            if drump_out:
                next_vertex=drump_out_local_critical(mesh=mesh,current_ring={next_vertex},name=name,slope_need=0,direction=direction)
            else:
                raise ValueError('vertex is local minimun')
    if mesh.vertex_attribute(key=next_vertex,name='boundary')!=0:
        print('end')
        return next_vertex
    else:
        return gradient_descent(mesh,next_vertex,name,direction,drump_out=drump_out)
        
def drump_out_local_critical(mesh:Mesh,current_ring,name='scalar_field',slope_need=0.1,direction=True,distances=None,start_vertex=None):
    """
    direction=True : local maximun
    """
    if distances is None:
        distances={}
        for vkey in current_ring:
            start_vertex=vkey
            distances[vkey]=0
            
    
    next_ring=set()
    selected_vertex=set()
    selected_vertex_slope={}
    ring_queue=deque(current_ring)

    for vkey in ring_queue:
        for neighbor in mesh.vertex_neighbors(vkey):
            new_distance=mesh.edge_length(vkey,neighbor)+distances[vkey]
            if neighbor not in distances or new_distance<distances[neighbor]:
                distances[neighbor]=new_distance
                next_ring.add(neighbor)
                slope=new_distance/(mesh.vertex_attribute(key=neighbor,name=name)-mesh.vertex_attribute(key=start_vertex,name=name))
                if slope>slope_need and direction or slope<-slope_need and not direction:
                    selected_vertex.add(neighbor)
                    selected_vertex_slope[neighbor]=abs(slope)
    
    if len(selected_vertex)!=0:

        return max(selected_vertex_slope, key=selected_vertex_slope.get)
    else:
    
        return drump_out_local_critical(mesh,selected_vertex,name,slope_need,direction,distances,start_vertex)
def kill_local_criticals(mesh:Mesh,maxim,minim,name='scalar_field',slope_need=0.1):
    for point in maxim:
        kill_local_critical(mesh,point,name,slope_need,True)
    for point in minim:
        kill_local_critical(mesh,point,name,slope_need,False)

def kill_local_critical(mesh:Mesh,current_ring,name='scalar_field',slope_need=0.1,direction=True,distances=None,start_vertex=None):
    """
    direction=True : local maximun
    """
    
    if not current_ring:
        raise ValueError('no point')
    if distances is None:
        print('killing',current_ring)
        distances={}
        
        start_vertex=current_ring
        distances[current_ring]=0
        current_ring={current_ring}
            
    
    next_ring=set()
    selected_vertex=set()
    selected_vertex_slope={}
    ring_queue=deque(current_ring)
    while ring_queue :
        vkey=ring_queue.popleft()
        touch_boundary=False
        for neighbor in mesh.vertex_neighbors(vkey):
            if name != 'z' or name != 'zs':
                new_distance=mesh.edge_length(vkey,neighbor)+distances[vkey]
            else:
                new_distance=((mesh.vertex[vkey]['x']-mesh.vertex[neighbor]['x'])**2+(mesh.vertex[vkey]['y']-mesh.vertex[neighbor]['y'])**2)**0.5+distances[vkey]
            if neighbor not in distances or new_distance<distances[neighbor]:
                #print(mesh.vertex[neighbor])
                if mesh.vertex_attribute(key=neighbor,name='boundary') != 0:
                    touch_boundary=True
                if neighbor in current_ring:
                    ring_queue.append( neighbor)

                distances[neighbor]=new_distance
                next_ring.add(neighbor)
                next_ring.discard(vkey)
                if name == 'zs':
                    slope=(mesh.vertex_attribute(key=neighbor,name='z')-mesh.vertex_attribute(key=start_vertex,name='z'))/((mesh.vertex_attribute(key=neighbor,name='x')-mesh.vertex_attribute(key=start_vertex,name='x'))**2+(mesh.vertex_attribute(key=neighbor,name='y')-mesh.vertex_attribute(key=start_vertex,name='y'))**2)**0.5
                    slope_true = (mesh.vertex_attribute(key=neighbor,name='z')-mesh.vertex_attribute(key=start_vertex,name='z'))/new_distance
                else:
                    slope=(mesh.vertex_attribute(key=neighbor,name=name)-mesh.vertex_attribute(key=start_vertex,name=name))/new_distance
                #print(neighbor,selected_vertex,slope)
                if (slope>slope_need and direction) or (slope<-slope_need and not direction):
                    
                    selected_vertex.add(neighbor)
                    if name == 'zs':
                        selected_vertex_slope[neighbor]=abs(slope_true)
                    else:
                        selected_vertex_slope[neighbor]=abs(slope)
    
    if len(selected_vertex)!=0:
        end_vertex=max(selected_vertex_slope, key=selected_vertex_slope.get)
        max_slope=selected_vertex_slope[end_vertex]
        if not direction:
            max_slope=-max_slope
        print('max_slope',max_slope,selected_vertex,selected_vertex_slope,direction)
        
       
        path=gradient_descent_path(mesh,end_vertex,distances,False)
        print('path',path)
        if len(path)<3:
            print(mesh.vertex_attribute(key=path[1],name=name),mesh.vertex_attribute(key=path[0],name=name),distances[path[0]])
            if touch_boundary:
                raise ValueError('touch boundary')
            raise Exception('imposible path')
        for pathvi,path_v in enumerate(path):
            #if pathvi==0:
            #    print(start_vertex,'startvertx',max_slope,(mesh.vertex[path_v][name]-mesh.vertex[start_vertex][name]),max_slope*distances[path_v])
            if name == 'zs':
                org_sf=mesh.vertex[path_v]['z']
                mesh.vertex[path_v]['z']=mesh.vertex[start_vertex]['z']+max_slope*distances[path_v]  
                #print(path_v,'changing,',org_sf-mesh.vertex[path_v]['z'],'  scalar field changeing from',org_sf,'to',mesh.vertex[path_v]['z'])          
            else:
                org_sf=mesh.vertex[path_v][name]
                mesh.vertex[path_v][name]=mesh.vertex[start_vertex][name]+max_slope*distances[path_v]
            




        return path
    else:
        if len(next_ring)==0:
            print(vkey,distances[vkey],[(neighbor,distances[neighbor]) for neighbor in mesh.vertex_neighbors(vkey)])
            from compas.files import OBJWriter 
            import os
            obj_writer = OBJWriter(filepath= os.path.join("C:\Differential", "edited_mesh.obj"), meshes=[mesh])
            obj_writer.write()
            raise ValueError('Can not find a path possible')
        return kill_local_critical(mesh,next_ring,name,slope_need,direction,distances,start_vertex)

def kill_minmal_with_heat_acumulate(mesh:Mesh,point,name='scalar_field',slope_need=0.001,direction=True):
    from gradient_optimization import heat_accumulater
    small_heat_accumulater=heat_accumulater(mesh,{vkey:mesh.vertex_attribute(key=vkey,name=name) for vkey in mesh.vertices()})
    
    small_heat_accumulater.accumulate_one_vertex_area(point,True)
    
   



def gradient_descent_path(mesh:Mesh,vertex,field,direction=True,pathvertices=None          ):
    '''
    direction=True : going up
    '''
    #print(vertex,pathvertices)
    if pathvertices is None:
        pathvertices=[]
    pathvertices.append(vertex)
    stop=False
    #print('gradient_descent',vertex)
    neighbors = mesh.vertex_neighbors(vertex)
    neighbors[:] = [item for item in neighbors if item in field]
    # 使用列表推导式找到 name 值最小的 neighbor
    if direction:
        next_vertex = max(neighbors, key=lambda neighbor:field[neighbor])
        if field[next_vertex] < field[vertex]:
            stop=True
     
    else:
        next_vertex = min(neighbors, key=lambda neighbor: field[neighbor])
        if field[next_vertex] > field[vertex]:
            stop=True
    if mesh.vertex_attribute(key=next_vertex,name='boundary')!=0:
        raise ValueError('cannot get out of boundary') 
    elif stop:
        print('end',pathvertices[0],pathvertices[-1])
        return pathvertices
    else:
        return gradient_descent_path(mesh,next_vertex,field,direction,pathvertices)    

def gradient_descent_distance(mesh:Mesh,vertex,guide_field,descent_distances,pathvertices=None):
    """
    This function should be run on a field without any local maximun or minimun
    going up on guide field
    """
    if pathvertices is None:
        pathvertices=[]
        if vertex in descent_distances:
            raise ValueError('vertex already in descent_distances')
            
    pathvertices.append(vertex)
    if vertex in descent_distances:
        return give_descent_distances_to_path(mesh,pathvertices,descent_distances)



    neighbors = mesh.vertex_neighbors(vertex)
    next_vertex = max(neighbors, key=lambda neighbor:guide_field[neighbor])
    return gradient_descent_distance(mesh,next_vertex,guide_field,descent_distances,pathvertices)
   

def give_descent_distances_to_path(mesh:Mesh,pathvertices:list,descent_distances):
    pathvertices.reverse()
    for i in range(len(pathvertices)-1):
        descent_distances[pathvertices[i]]=descent_distances[pathvertices[i-1]]+mesh.edge_length(pathvertices[i-1],pathvertices[i])
    return descent_distances[pathvertices[-1]]

def gradient_descent_distances_with_related_boundary(mesh:Mesh,guide_field,descent_distances_going_up,related_boundaries_High,descent_distances_going_down,related_boundaries_Low):

    sorted_vertices=sorted(mesh.vertices(),key=lambda vertex:guide_field[vertex])
    for vertex in sorted_vertices:
        if vertex not in descent_distances_going_up:
            gradient_descent_distance_with_related_boundary(mesh,vertex,guide_field,descent_distances_going_up,related_boundaries_High)
        if vertex not in descent_distances_going_down:
            gradient_descent_distance_with_related_boundary_flip(mesh,vertex,guide_field,descent_distances_going_down,related_boundaries_Low)
    return descent_distances_going_up,related_boundaries_High,descent_distances_going_down,related_boundaries_Low
def gradient_descent_distance_with_related_boundary(mesh:Mesh,vertex,guide_field,descent_distances,related_boundaries,pathvertices=None):
    """
    This function should be run on a field without any local maximun or minimun
    going up on guide field
    """
    
    # if pathvertices is not None and vertex in pathvertices:
    #     print([guide_field[n] for n in mesh.vertex_neighbors(vertex)])
    if pathvertices is None:
        pathvertices=[]
        if vertex in descent_distances:
            if vertex in related_boundaries:
                raise ValueError('vertex already in descent_distances and boundary')
            else:
                raise ValueError('vertex already in descent_distances but not in boundary')
        elif vertex in related_boundaries:
            raise ValueError('vertex already in boundary but not in descent_distances')

            
    pathvertices.append(vertex)
    if vertex in descent_distances:
        return give_descent_distances_and_related_boundary_to_path(mesh,pathvertices,descent_distances,related_boundaries)



    neighbors = mesh.vertex_neighbors(vertex)
    next_vertex = max(neighbors, key=lambda neighbor:guide_field[neighbor])
    return gradient_descent_distance_with_related_boundary(mesh,next_vertex,guide_field,descent_distances,related_boundaries,pathvertices)

def gradient_descent_distance_with_related_boundary_flip(mesh:Mesh,vertex,guide_field,descent_distances,related_boundaries,pathvertices=None):
    """
    This function should be run on a field without any local maximun or minimun
    going up on guide field
    """
    if pathvertices is None:
        pathvertices=[]
        if vertex in descent_distances:
            if vertex in related_boundaries:
                raise ValueError('vertex already in descent_distances and boundary')
            else:
                raise ValueError('vertex already in descent_distances but not in boundary')
        elif vertex in related_boundaries:
            raise ValueError('vertex already in boundary but not in descent_distances')

            
    pathvertices.append(vertex)
    if vertex in descent_distances:
        return give_descent_distances_and_related_boundary_to_path(mesh,pathvertices,descent_distances,related_boundaries)



    neighbors = mesh.vertex_neighbors(vertex)
    next_vertex = min(neighbors, key=lambda neighbor:guide_field[neighbor])
    return gradient_descent_distance_with_related_boundary_flip(mesh,next_vertex,guide_field,descent_distances,related_boundaries,pathvertices)
   

def give_descent_distances_and_related_boundary_to_path(mesh:Mesh,pathvertices:list,descent_distances,related_boundaries):
    pathvertices.reverse()
    for i in range(1,len(pathvertices)):
        descent_distances[pathvertices[i]]=descent_distances[pathvertices[i-1]]+mesh.edge_length(pathvertices[i-1],pathvertices[i])
        related_boundaries[pathvertices[i]]=related_boundaries[pathvertices[i-1]]
        
    return descent_distances[pathvertices[-1]]


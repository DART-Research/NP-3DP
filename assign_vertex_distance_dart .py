import logging
from compas_slicer.pre_processing.preprocessing_utils import blend_union_list, stairs_union_list, chamfer_union_list,CompoundTarget
from compas_slicer.utilities.utils import remap_unbound
import numpy as np
from compas.datastructures import Mesh
from compound_target_dart import CompoundTargetDart
from gradient_descent import gradient_descent_distances_with_related_boundary

logger = logging.getLogger('logger')

__all__ = ['assign_interpolation_distance_to_mesh_vertices',
           'assign_interpolation_distance_to_mesh_vertex',
           'assign_final_distance_to_mesh_vertices',
           'assign_z_distance_to_mesh_vertices',
           'get_final_distance',
           'assign_up_distance_to_mesh_vertices',
           'assign_multi_distance_to_mesh_vertices',
           'assign_gradient_descent_distance_to_mesh_vertices']

def assign_interpolation_distance_to_mesh_vertices(mesh:Mesh, weight, target_LOW=None, target_HIGH=None):
    """
    Fills in the 'get_distance' attribute of every vertex of the mesh.

    Parameters
    ----------
    mesh: :class: 'compas.datastructures.Mesh'
    weight: float,
        The weighting of the distances from the lower and the upper target, from 0 to 1.
    target_LOW: :class: 'compas_slicer.pre_processing.CompoundTarget'
        The lower compound target.
    target_HIGH:  :class: 'compas_slicer.pre_processing.CompoundTarget'
        The upper compound target.
    """
    for i, vkey in enumerate(mesh.vertices()):
        d = assign_interpolation_distance_to_mesh_vertex(vkey, weight, target_LOW, target_HIGH)
        mesh.vertex[vkey]['scalar_field'] = d




def assign_interpolation_distance_to_mesh_vertex(vkey, weight, target_LOW=None, target_HIGH=None):
    """
    Fills in the 'get_distance' attribute for a single vertex with vkey.

    Parameters
    ----------
    vkey: int
        The vertex key.
    weight: float,
        The weighting of the distances from the lower and the upper target, from 0 to 1.
    target_LOW: :class: 'compas_slicer.pre_processing.CompoundTarget'
        The lower compound target.
    target_HIGH:  :class: 'compas_slicer.pre_processing.CompoundTarget'
        The upper compound target.
    """
    if target_LOW is not None and target_HIGH is not None:  # then interpolate targets
        d = get_weighted_distance(vkey, weight, target_LOW, target_HIGH)
    elif target_LOW is not None:  # then offset target
        #print("target Low")
        #offset = weight * target_LOW.get_max_dist()
        d = target_LOW.get_all_distances_for_vkey(vkey,True) 
    elif target_HIGH is not None:
        #print('target_high')
        d=target_HIGH.get_all_distances_for_vkey(vkey,True)
    else:
        raise ValueError('You need to provide at least one target')
    return d

def assign_final_distance_to_mesh_vertices(mesh:Mesh, weight, target_LOW:CompoundTargetDart, target_HIGH:CompoundTargetDart):
    for vkey in mesh.vertices():
        d = assign_final_distance_to_mesh_vertice(vkey,weight,target_LOW,target_HIGH)
        mesh.vertex[vkey]['scalar_field']=d
def assign_multi_distance_to_mesh_vertices(mesh:Mesh, weight, target_LOW:CompoundTargetDart, target_HIGH:CompoundTargetDart,inter):   
    distances_l=[]
    for vkey in mesh.vertices():   
        distance_=target_LOW.get_offset_distances_veky(vkey)
        distances_l.append(distance_)
    max_dist=max(distances_l)

    for vkey in mesh.vertices():
        # if vkey>1:
        #     print("after ",mesh.vertex[vkey-1]['scalar_field'])
        d2=distances_l[vkey]/max_dist
        d1=assign_final_distance_to_mesh_vertice(vkey,weight,target_LOW,target_HIGH)
        d=inter*d1+d2*(1-inter)
        mesh.vertex[vkey]['scalar_field']=d
        # print("org ",mesh.vertex[vkey]['scalar_field'])


def assign_final_distance_to_mesh_vertice(vkey, weight, target_LOW:CompoundTargetDart, target_HIGH:CompoundTargetDart):
    if target_LOW and target_HIGH:  # then interpolate targets
        d = get_final_distance(vkey,  target_LOW, target_HIGH)
    elif target_LOW:  # then offset target
        print("target Low")
        offset = weight * target_LOW.get_max_dist()
        d = target_LOW.get_distance(vkey) - offset
    else:
        raise ValueError('You need to provide at least one target')
    return d

def get_final_distance(vkey,  target_LOW:CompoundTargetDart, target_HIGH:CompoundTargetDart):
    d_low=target_LOW.get_offset_distances_veky(vkey)
    d_high=target_HIGH.get_offset_distances_veky(vkey)
    together=(d_high+d_low)
    if together==0:
        
        print(vkey,d_low,d_high,"!!!!!!!!!!!")
    return(d_low/together)


def assign_org_gra_distance_to_mesh_vertices(mesh:Mesh,target_LOW:CompoundTargetDart,target_HIGH:CompoundTargetDart):
    for vkey in mesh.vertices():
        d = assign_org_gra_distance_to_mesh_vertice(vkey,target_LOW,target_HIGH)
        mesh.vertex[vkey]['scalar_field']=d

def assign_gradient_descent_distance_to_mesh_vertices(mesh:Mesh,target_LOW:CompoundTargetDart,target_HIGH:CompoundTargetDart,guide_field):
    descent_distances_going_up,related_boundaries_High,descent_distances_going_down,related_boundaries_Low={},{},{},{}
    for bi,boundary in enumerate(target_HIGH.clustered_vkeys):
        for vertex in boundary:
            descent_distances_going_up[vertex]=target_HIGH.get_offset_distances_veky(vertex)
            related_boundaries_High[vertex]=bi
    for bi,boundary in enumerate(target_LOW.clustered_vkeys):
        for vertex in boundary:
            descent_distances_going_down[vertex]=target_LOW.get_offset_distances_veky(vertex)
            related_boundaries_Low[vertex]=bi

    descent_distances_going_up,related_boundaries_High,descent_distances_going_down,related_boundaries_Low=gradient_descent_distances_with_related_boundary(mesh,guide_field,descent_distances_going_up,related_boundaries_High,descent_distances_going_down,related_boundaries_Low)
    for vkey in mesh.vertices():
        d = descent_distances_going_down[vkey]/(descent_distances_going_down[vkey]+descent_distances_going_up[vkey])
        mesh.vertex[vkey]['scalar_field']=d

def assign_org_gra_distance_to_mesh_vertice(vkey,target_LOW:CompoundTarget,target_HIGH:CompoundTarget):
    """
    Fills in the 'get_distance' attribute for a single vertex with vkey.

    Parameters
    ----------
    vkey: int
        The vertex key.
    weight: float,
        The weighting ofthe distances from the lower and the upper target, from 0 to 1
    """
    if target_LOW and target_HIGH:
        d_low = target_LOW.get_distance(vkey)  # float
        ds_high = target_HIGH.get_all_distances_for_vkey(vkey)  # list of floats (# number_of_boundaries)
        weight_maxs=target_HIGH.weight_max_per_cluster
        # if target_HIGH.number_of_boundaries > 1:
        #     weights_remapped = [remap_unbound(weight, 0, weight_max, 0, 1)
        #                         for weight_max in target_HIGH.weight_max_per_cluster]
        #     weights = weights_remapped
        # else:
        #     weights = [weight]

        distances = [d_low*weight_max/(d_low+d_high) for d_high, weight_max in zip(ds_high, weight_maxs)]

        # return the distance based on the union method of the high target
        if target_HIGH.union_method == 'min':
            # --- simple union
            #H_distances=[(distance/weight_max-1)*d_low+(distance/weight_max)*d_high for distance,weight_max,d_high in zip(distances,weight_maxs,ds_high)]
            #offseted_ds_high=[d_high*(1-weight_max) for d_high,weight_max in zip(ds_high,weight_maxs)]
            #return distances[offseted_ds_high.index(np.min(offseted_ds_high))]
            return np.max(distances)


    # --- simple calculation (without uneven weights)
    else:
        print("else__________assign org_gra")
        d_low = target_LOW.get_distance(vkey)
        d_high = target_HIGH.get_distance(vkey)
        return (d_low/(d_low+d_high))        
    

def assign_z_distance_to_mesh_vertices(mesh:Mesh, weight, target_LOW:CompoundTargetDart, target_HIGH:CompoundTargetDart):
    for vkey in mesh.vertices():
        d = mesh.vertex_attribute(key=vkey,name='z')
        mesh.vertex[vkey]['scalar_field']=d

def assign_up_distance_to_mesh_vertices(mesh:Mesh,target_HIGH:CompoundTargetDart,target_index):
    """
    to get the distance field of single target curve
    """
    distances_l=[]
    for vkey in mesh.vertices():
        if target_index>=0:
            distance_=target_HIGH.get_all_distances_for_vkey(i=vkey)[target_index]
        else: 
            distance_=target_HIGH.get_offset_distances_veky(vkey)
        distances_l.append(distance_)
    max_dist=max(distances_l)

    for vkey in mesh.vertices():
        if vkey==1:
            print(mesh.vertex[0]['scalar_field'])
        d=distances_l[vkey]/max_dist
        #d=distances_l[veky]
        mesh.vertex[vkey]['scalar_field']=d
        if vkey==0:
            print(mesh.vertex[vkey]['scalar_field'])
            







def get_weighted_distance(vkey, weight, target_LOW:CompoundTargetDart, target_HIGH:CompoundTargetDart):
    """
    Computes the weighted get_distance for a single vertex with vkey.

    Parameters
    ----------
    vkey: int
        The vertex key.
    weight: float,
        The weighting of the distances from the lower and the upper target, from 0 to 1.
    target_LOW: :class: 'compas_slicer.pre_processing.CompoundTarget'
        The lower compound target.
    target_HIGH:  :class: 'compas_slicer.pre_processing.CompoundTarget'
        The upper compound target.
    """
    # NOTE --- Yichuan's algorthm of offset the base
    d_low=target_LOW.get_offset_distances_veky(vkey)
    d_high=target_HIGH.get_offset_distances_veky(vkey)
    # NOTE print("try",d_low,"and",d_high)
    return (weight - 1) * d_low + weight * d_high
### end
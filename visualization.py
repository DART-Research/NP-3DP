import compas_slicer_ghpython.visualization as gh_compas_slicer

import os
import json
import rhinoscriptsyntax as rs
from compas.datastructures import Mesh
import Rhino.Geometry as rg
import ghpythonlib.components as gh
from compas_ghpython.artists import MeshArtist
from compas.geometry import Frame
from compas_ghpython.utilities import list_to_ghtree
import math
def load_slicer_names(load_path,folder_name):
    data=load_json_file(load_path, folder_name, 'slicer_names.json')
    return data

def load_mesh_topo(load_path,folder_name):
    
    data=load_json_file(load_path, folder_name, 'mesh_topo.json')
    ups=[]
    downs=[]
    dist_up=[]
    dist_low=[]
    path_point=[]
    for i in data:
        try:
            up=i['lower_layer']            
            ups.append(up[0]+0.3*up[1]) 
   
        except:

            ups.append(-1)
            
        try:
            up_dist=i['lower_dist']  
            dist_up.append(up_dist)  
        except:
            print(i,'du')
            dist_up.append(-1)
        try:
            down=i['upper_layer']
            downs.append(down[0]+0.3*down[1])

        except:

            downs.append(-1)

        try:
            low_dist=i['upper_dist']
            dist_low.append(low_dist)    
        except:
            print(i,'dl')
            dist_low.append(-1)    

  
 
    return ups,downs,dist_up,dist_low

def load_polyline_list(load_path,folder_name,json_name):
    path_save=os.path.join(load_path, folder_name)
    if os.path.exists(path_save)  :
        data=load_json_file(load_path,folder_name,json_name)
        print(data)
        paths=[]
        for polyline in data:
            pts=[]
            for point in polyline:
                pt=gh.ConstructPoint(point[0], point[1], point[2])
                pts.append(pt)
            gh_polyline = gh.PolyLine(pts,False)
            paths.append(gh_polyline)
        return paths
    else:
        print('No layers have been saved in the json file. Is this the correct json?')
        print(path_save)


def load_slicer_list(load_path,folder_name,json_name,gradient_max=0.001,gradient_min=0.0001,gradient_angle_min=math.pi/10,return_points=False):
    """ Loads slicer data. """
    path_save=os.path.join(load_path, folder_name,'output','sorted_paths')
    if os.path.exists(path_save) and os.path.isdir(path_save) and not return_points:
        data=load_json_file(path_save, '', json_name)
        paths_nested_list=[]
        if data:
            for layer in data:
                paths_nested_list.append([])
                for path_0ss in layer:
                    paths_nested_list[-1].append([])
                    paths_nested_list[-1].append([])
                    good_path_0s = path_0ss[0]
                    bad_path_0s = path_0ss[1]
                    for good_path_0 in good_path_0s:
                        pts = []
                        for point_0 in good_path_0:
                            pt=gh.ConstructPoint(point_0[0], point_0[1], point_0[2])
                            pts.append(pt)
                        paths_nested_list[-1][0].append(gh.PolyLine(pts,False))
                    for bad_path_0 in bad_path_0s:
                        pts = []
                        for point_0 in bad_path_0:
                            pt=gh.ConstructPoint(point_0[0], point_0[1], point_0[2])
                            pts.append(pt)
                        paths_nested_list[-1][1].append(gh.PolyLine(pts,False))
        return paths_nested_list

    else:
        data = load_json_file(load_path, folder_name, json_name)

        paths_nested_list = []
        paths_points_nested_list = []   
        points_nested_list =[]
        gradients_nested_list = []
        are_closed = []
        all_points = []

        if data:
            if 'layers' in data:
                print("layer")

                layers_data = data['layers']

                for i in range(len(layers_data)):
                    paths_nested_list.append([])  # save each layer on a different list
                    paths_points_nested_list.append([])
                    gradients_nested_list.append([])
                    points_nested_list.append([])
                    layer_data = layers_data[str(i)]
                    paths_data = layer_data['paths']


                        
                        
                    if paths_data[str(0)]['gradients']:

                        for j in range(len(paths_data)):
                            #print(i,j)
                            path_data = paths_data[str(j)]                    
                            paths_nested_list[-1].append([])
                            paths_nested_list[-1].append([])
                            paths_points_nested_list[-1].append([])
                            paths_points_nested_list[-1].append([])

                            print("gradient found")
                            pts_good=[]
                            pts_bad=[]
                            if_good=None
    
                            if len(path_data['points']) > 2:  # ignore smaller curves that throw errors
                                for k in range(len(path_data['points'])):
                                    if k==0:
                                        pt0=path_data['points'][str(k)]
                                        pt0 = gh.ConstructPoint(pt0[0], pt0[1], pt0[2])  
                                    pt_o = path_data['points'][str(k)]
                                    try:
                                        gr = path_data['gradients'][str(k)]
                                    except:
                                        print("error",k)
                                        gr = path_data['gradients'][str(0)]  
                                    points_nested_list[-1].append( gh.ConstructPoint(pt_o[0], pt_o[1], pt_o[2]))   
                                    gradients_nested_list[-1].append( gh.VectorXYZ(gr[0], gr[1], gr[2])[0]) 
                                    gr_norm=list_norm(gr)
                                    gr_angle=math.atan(gr[2]/list_norm(gr[0:2]))     
                                    if gr_norm > gradient_min and gr_norm < gradient_max and gr_angle > gradient_angle_min:
                                        if  if_good is None:
                                            if_good=True
                                            pts_good.append([])
                                            
                                        elif if_good==False:
                                            if_good=True
                                            pts_good.append([])
                                            print(pt)
                                            pts_good[-1].append(pt)
                                        pt = gh.ConstructPoint(pt_o[0], pt_o[1], pt_o[2])  
                                        pts_good[-1].append(pt)  
                                    else:
                                        if if_good is None:
                                            if_good=False
                                            pts_bad.append([])
                                        elif if_good==True:
                                            if_good=False
                                            pts_bad.append([])
                                            print(pt)
                                            pts_bad[-1].append(pt)                                        
                                        pt = gh.ConstructPoint(pt_o[0], pt_o[1], pt_o[2])   
                                        pts_bad[-1].append(pt)  
            
                                
                                    
                                
                                if path_data['is_closed']:
                            
                                    if if_good:
                                        pts_good[-1].append(pt0)
                                    else:
                                        pts_bad[-1].append(pt0)
                                print(i,j,len(pts_good),len(pts_bad))
                                for path_pts in pts_good:
                                    print(len(path_pts))
                                    path = gh.PolyLine(path_pts)
                                    #print(path)
                                    paths_nested_list[-1][0].append(path)  
                                    paths_points_nested_list[-1][0].append(path_pts)
                                for path_pts in pts_bad:
                                    print(len(path_pts))
                                    path = gh.PolyLine(path_pts)
                                    #print(path)
                                    paths_nested_list[-1][1].append(path)  
                                    paths_points_nested_list[-1][1].append(path_pts)                                                  
                    else:
                        for j in range(len(paths_data)):
                            #print(i,j)
                            path_data = paths_data[str(j)]  
                            pts = []
                            if len(path_data['points']) > 2:  # ignore smaller curves that throw errors
                                for k in range(len(path_data['points'])):
                                    pt = path_data['points'][str(k)]
                                    pt = rs.AddPoint(pt[0], pt[1], pt[2])  # re-create points
                                    pts.append(pt)
                                all_points.extend(pts)
                                if path_data['is_closed']:
                                    pts.append(pts[0])
                                path = rs.AddPolyline(pts,False)
                                #print(path)
                                paths_nested_list[-1].append(path)
                                paths_points_nested_list[-1].append(pts)

            else:
                print('No layers have been saved in the json file. Is this the correct json?')

        print('The slicer contains %d layers. ' % len(paths_nested_list))
        
    
        save_nested_list(path_save,json_name,paths_points_nested_list)

    if return_points:
        return paths_nested_list, points_nested_list,gradients_nested_list
    else:
        return paths_nested_list

def load_gradient_tangent(path,folder_name,json_name):
    """
    not finished yet
    """

    data = load_json_file(path, folder_name, json_name)
    gradient_tangents = []
    
    if data:
        data_keys=sorted([int(key) for key in data.keys()])
        for vkey in data_keys:
            print(vkey)
            gradient=data[str(vkey)]
            #print(gradient)
            gradient_tangent = math.atan(gradient[2]/(gradient[0]**2+gradient[1]**2)**0.5)
            #print(gradient_tangent)    
            gradient_tangents.append(gradient_tangent)
    return gradient_tangents

def load_gradient(path,folder_name,json_name):
    """
    not finished yet
    """
    data = load_json_file(path, folder_name, json_name)
    gradient_tangents = []
    if data:
        data_keys=sorted([int(key) for key in data.keys()])
        for vkey in data_keys:
         
            gradient=data[str(vkey)]
            #print(gradient)
            gradient_tangent = gh.VectorXYZ(gradient[0],gradient[1],gradient[2])[0]
            #print(gradient_tangent)    
            gradient_tangents.append(gradient_tangent)
    return (gradient_tangents)
def list_norm(list):
    sum=0
    for i in list:
        sum+=i**2
    return math.sqrt(sum)

def load_slicer_tree(path,folder_name,json_name,gradient_max=0.001,gradient_min=0.0001,gradient_angle_min=math.pi/10,return_points=False):
    if return_points:
        paths,points,gradients=load_slicer_list(path,folder_name,json_name,gradient_max,gradient_min,gradient_angle_min,True)
        paths=list_to_ghtree(paths)
        points=list_to_ghtree(points)
        gradients=list_to_ghtree(gradients)
        return paths,points,gradients
    else:
        paths=load_slicer_list(path,folder_name,json_name,gradient_max,gradient_min,gradient_angle_min,False)
        paths=list_to_ghtree(paths)
        return paths
def load_slicer_trees(path,folder_name,json_names,gradient_max=0.001,gradient_min=0.0001,gradient_angle_min=math.pi/10):
    path_trees=[]

    for name in json_names:
        paths=load_slicer_list(path,folder_name,name,gradient_max,gradient_min,gradient_angle_min,False)
        print(len(paths))
        path_trees.append(paths)
    path_trees=list_to_ghtree(path_trees)
    return path_trees


def load_nested_influence(path,folder_name,json_names):
    data=[]
    for name in json_names:
        influences=load_json_file(path,folder_name,name)
        influences=[list(row) for row in zip(*influences)]
        data.append(influences)
    nested_list=list_to_ghtree(data)
    print(len(data),len(data[0]))
    print(nested_list.Branches)
    return nested_list

def load_nested_data(path,folder_name,json_names):
    data=[]
    for name in json_names:
        influences=load_json_file(path,folder_name,name)
        
        
        data.append(influences)
    
    nested_list=list_to_ghtree(data)
    print(len(data),len(data[0]))
    print(nested_list.Branches)
    return nested_list
def load_influence(path,folder_name,json_name):
    data = load_json_file(path, folder_name, json_name)
    nested_list = list_to_ghtree(data)
    return nested_list



# def load_slicer(path, folder_name, json_name):
#     """ Loads slicer data. """
#     data = load_json_file(path, folder_name, json_name)

#     mesh = None
#     paths_nested_list = []
#     are_closed = []
#     all_points = []

#     if data:

#         if 'mesh' in data:
#             compas_mesh = Mesh.from_data(data['mesh'])
#             artist = MeshArtist(compas_mesh)
#             artist.show_mesh = True
#             artist.show_vertices = False
#             artist.show_edges = False
#             artist.show_faces = False
#             mesh = artist.draw()
#         else:
#             print('No mesh has been saved in the json file.')

#         if 'layers' in data:

#             layers_data = data['layers']

#             for i in range(len(layers_data)):
#                 paths_nested_list.append([])  # save each layer on a different list
#                 layer_data = layers_data[str(i)]
#                 paths_data = layer_data['paths']

#                 for j in range(len(paths_data)):
#                     print(i,j)
#                     path_data = paths_data[str(j)]
#                     pts = []

#                     are_closed.append(path_data['is_closed'])

#                     if len(path_data['points']) > 2:  # ignore smaller curves that throw errors
#                         for k in range(len(path_data['points'])):
#                             pt = path_data['points'][str(k)]
#                             pt = rs.AddPoint(pt[0], pt[1], pt[2])  # re-create points
#                             pts.append(pt)
#                         all_points.extend(pts)
#                         # The following two lines are added by Yichuan Li
#                         if path_data['is_closed']:
#                             pts.append(pts[0])
#                         path = rs.AddPolyline(pts)
#                         paths_nested_list[-1].append(path)

#         else:
#             print('No layers have been saved in the json file. Is this the correct json?')

#     print('The slicer contains %d layers. ' % len(paths_nested_list))
#     paths_nested_list = list_to_ghtree(paths_nested_list)
#     return mesh, paths_nested_list, are_closed, all_points

def load_json_file(path, folder_name, json_name, in_output_folder=True):
    """ Loads data from json. """

    if in_output_folder:
        filename = os.path.join(os.path.join(path), folder_name, 'output', json_name)
    else:
        filename = os.path.join(os.path.join(path), folder_name, json_name)
    data = None

    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        print("Loaded Json: '" + filename + "'")
    else:
        print("Attention! Filename: '" + filename + "' does not exist. ")

    return data

def save_nested_list(file_path, file_name, nested_list):
    """
    
    
    :param file_path: 
    :param file_name:
    :param nested_list:
    """
    nested_list=convert_to_serializable(nested_list)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    full_path = os.path.join(file_path, file_name)
    
    
    with open(full_path, 'w') as file:
        json.dump(nested_list, file)
    print("save_nested_list_",file_name," to ",file_path)

def is_point3d(obj):
    return isinstance(obj, rg.Point3d)

def convert_to_serializable(obj):

    if is_point3d(obj):
        return [obj.X, obj.Y, obj.Z]
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def serialize_nested_list(nested_list):

    return convert_to_serializable(nested_list)
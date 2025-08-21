import logging
import compas_slicer
import compas_slicer.utilities.utils as utils
import numpy as np
from compas_slicer.geometry import Path,layer,VerticalLayersManager,VerticalLayer
from typing import List,Tuple,Set
import copy
from compas.geometry import Point,Vector

class path_dart(Path):
    def __init__(self, points:Point, is_closed,gradients=None,edges=None,tt=None,from_dict=False):
        if from_dict:
            self.points=points
            self.is_closed=is_closed
        
        else:
            super().__init__(points, is_closed)
            self.points=[]
            self.number_of_points=len(points)
            if gradients is not None and gradients:
                self.has_gradients=True
                
                self.gradients=gradients
                for pt,gradiet in zip(points,gradients):
                    self.points.append({'point':pt,'gradient':gradiet})
            else:
                self.has_gradients=False

                for i in range(len(points)):
                    pt=points[i]
                    self.points.append({'point':pt})
        
        
            if edges is not None:
                self.faces_up=set()
                self.faces_down=set()
                for i,point_i in enumerate(self.points):
                    point_i['edge']=edges[i]
                    point_i['t']=tt[i]
        
        #print(len(self.points),len(self.gradients))
        #print(self.points[:10],self.gradients[:10])
    def to_data(self,whole_points_data=False):
        """Returns a dictionary of structured data representing the data structure.

        Returns
        -------
        dict
            The path's data.

        """
        if whole_points_data:
            points=copy.deepcopy(self.points)
            for point in points:
                point['point']=point['point'].to_data()
            data = {'points': points,
                    'is_closed': self.is_closed} 
        elif self.has_gradients: 
        
            data = {'points': {i: point.to_data() for i, point in enumerate(self.get_points())},
                    'is_closed': self.is_closed,'gradients': {i: arry.tolist() for i, arry in enumerate(self.get_point_gradients())}} 
        else:           
            data = {'points': {i: point.to_data() for i, point in enumerate(self.get_points())},
                    'is_closed': self.is_closed}
        return data
    @classmethod
    def from_dict(cls, data:dict):
        points=data['points']
        for point in points:
            point['point']=Point.from_data(point['point'])
        is_closed=data['is_closed']
        return cls(points,is_closed,from_dict=True)


    def get_whole_points_data(self):
        return self.points
    def get_points(self)->List[Point]:
        return [point['point'] for point in self.points]
    def get_point_gradients(self)->List[np.ndarray]:
        return [point['gradient'] for point in self.points] 
    def get_whole_point_data(self,i)->dict:
        return self.points[i]
    def get_edges(self)->List[Tuple[int, int]]:
        return [point['edge'] for point in self.points]
    def change_edge(self,i,edge,t):
        self.points[i]['edge']=edge
        self.points[i]['t']=t
    def check_edge(self):
        for i in range(self.number_of_points):

            if not 'edge' in self.get_whole_point_data(i).keys():
                return False
        return True
    def get_t(self)->List[float]:
        return [point['t'] for point in self.points]
    
    def apply_w_key(self,key,point_id):
        self.points[point_id]['key']=key
    
    def get_keys(self)->List[int]:
        return [point['key'] for point in self.points]
    
    def add_faces(self,faces_up=set(),faces_down=set()):
        self.faces_up.update(faces_up)
        self.faces_down.update(faces_down)
    def add_edge_up_down(self,i,edge_up,edge_down):
        self.points[i]['edge_up_pt']=edge_up
        self.points[i]['edge_down_pt']=edge_down
    def get_edge_down_pt(self,i)->int:
        return self.points[i]['edge_down_pt']
    def get_edge_up_pt(self,i)->int:
        return self.points[i]['edge_up_pt']
    def update_faces(self,face_edit_data:dict):
        self.faces_up=expand_list_with_dict(self.faces_up,face_edit_data)
        self.faces_down=expand_list_with_dict(self.faces_down,face_edit_data)


    
def expand_list_with_dict(lst, d):
    updated = True

    new_list = []
    for item in lst:
        if item in d:  # If the item is a key in the dictionary
            new_list.extend(d[item])  # Replace with the corresponding values
               
        else:
            new_list.append(item)  # Keep the item as is
        lst = new_list  # Update the list for the next iteration
    return lst



  
class VerticalLayer_dart(VerticalLayer):
    """
    A Layer stores a group of ordered paths that are generated when a geometry is sliced.
    Layers are typically organized horizontally, but can also be organized vertically (see VerticalLayer).
    A Layer consists of one, or multiple Paths (depending on the geometry).

    Attributes
    ----------
    paths: list
        :class:`compas_slicer.geometry.Path`
    is_brim: bool
        True if this layer is a brim layer.
    number_of_brim_offsets: int
        The number of brim offsets this layer has (None if no brim).
    is_raft: bool
        True if this layer is a raft layer.
    """
    def __init__(self,id=0,paths=None):
        super().__init__(id,paths)
    def append_(self, path):
        """ Add path to self.paths list. """
        self.paths.append(path)
        self.compute_head_centroid()
        self.calculate_z_bounds()
    

    def compute_head_centroid(self):
        """ Find the centroid of all the points of the last path in the self.paths list"""
        #print(self.paths[-1].points[0])

        pts = np.array([point['point'] for point in self.paths[-1].points])
   
        self.head_centroid = np.mean(pts, axis=0)   
    def calculate_z_bounds(self):
        """ Fills in the attribute self.min_max_z_height. """
        assert len(self.paths) > 0, "You cannot calculate z_bounds because the list of paths is empty."
        z_min = 2 ** 32  # very big number
        z_max = -2 ** 32  # very small number
        for path in self.get_path():
            for pt_all in path.points:
                pt=pt_all['point']
                z_min = min(z_min, pt[2])
                z_max = max(z_max, pt[2])
        self.min_max_z_height = (z_min, z_max)
    def get_path(self)->List[path_dart]:
        return self.paths
    def remove_path(self,path:path_dart):
        self.paths.remove(path)
    
    def set_paths(self,paths:List[path_dart]):
        self.paths=paths
class  verticalLayerManager_dart:



    def __init__(self, threshold_max_centroid_dist=25.0, max_paths_per_layer=None):
        #super.__init__(threshold_max_centroid_dist, max_paths_per_layer)
        self.layers = [VerticalLayer_dart(id=0)]
        self.threshold_max_centroid_dist = threshold_max_centroid_dist
        self.max_paths_per_layer = max_paths_per_layer
    
    def add(self, path:path_dart):
        selected_layer = None

        #  Find an eligible layer for path (called selected_layer)
        if len(self.layers[0].paths) == 0:  # first path goes to first layer
            selected_layer = self.layers[0]

        else:  # find the candidate segment for new isocurve
            centroid = np.mean(np.array(path.points), axis=0)
            other_centroids = get_vertical_layers_centroids_list(self.layers)
            candidate_layer = self.layers[utils.get_closest_pt_index(centroid, other_centroids)]

            if np.linalg.norm(candidate_layer.head_centroid - centroid) < self.threshold_max_centroid_dist:
                if self.max_paths_per_layer:
                    if len(candidate_layer.paths) < self.max_paths_per_layer:
                        selected_layer = candidate_layer
                else:
                    selected_layer = candidate_layer

            if not selected_layer:  # then create new layer
                selected_layer = VerticalLayer_dart(id=self.layers[-1].id + 1)
                self.layers.append(selected_layer)

        selected_layer.append_(path)
    def add_with_layer(self,path:path_dart,layer_id):
        if len(self.layers)>layer_id:

            selected_layer = self.layers[layer_id]
        else:  # then create new layer
            selected_layer = VerticalLayer_dart(id=layer_id)

            self.layers.append(selected_layer)


        self.get_layer(layer_id).append_(path)
    def get_layer(self, layer_id)->VerticalLayer_dart:
        return self.layers[layer_id]
    @property
    def get_layer_count(self):
        return len(self.layers)

def get_vertical_layers_centroids_list(vert_layers):
    """ Returns a list with points that are the centroids of the heads of all vertical_layers_print_data. The head
    of a vertical_layer is its last path. """
    head_centroids = []
    for vert_layer in vert_layers:
        head_centroids.append(vert_layer.head_centroid)
    return head_centroids
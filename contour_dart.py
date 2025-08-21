from compas_slicer.slicers.slice_utilities import ScalarFieldContours
from compas_slicer.slicers.slice_utilities import ContoursBase
from compas.geometry import Vector, add_vectors, scale_vector
from compas.datastructures import Mesh
from layer_dart import verticalLayerManager_dart,path_dart
from compas_slicer.geometry import Path
from compas.geometry import Point, distance_point_point_sqrd
from gradient_evaluation_dart import GradientEvaluation_Dart
from compas_slicer.slicers.slice_utilities import create_graph_from_mesh_edges, sort_graph_connected_components

class ScalarFieldContours_layer(ScalarFieldContours):
    def __init__(self, mesh:Mesh, layer,gradient_evaluation:GradientEvaluation_Dart,edge_needed=None,discard_layer_num=0,output_edge=False):
        super().__init__(mesh)
        self.layer=layer
        self.edges_needed=edge_needed
        self.output_edge=output_edge
        if self.output_edge:
            self.sorted_t_clusters={}
        if gradient_evaluation is not None:
            self.check_gradient=True
            self.edge_gradient=gradient_evaluation.edge_gradient
        else:
            self.check_gradient=False
        self.sorted_point_gradient_clusters={}
        self.sorted_gradient_clusters={}
        self.discard_layer_num=discard_layer_num
    def compute(self):
        self.find_intersections()
        self.intersection_data_point={}
        for key in self.intersection_data:
            self.intersection_data_point[key]=self.intersection_data[key]['point']
        G = create_graph_from_mesh_edges(self.mesh, self.intersection_data_point, self.edge_to_index)
        sorted_indices_dict = sort_graph_connected_components(G)

        nodeDict = dict(G.nodes(data=True))
        for key in sorted_indices_dict:
            sorted_indices = sorted_indices_dict[key]
            self.sorted_edge_clusters[key] = [nodeDict[node_index]['mesh_edge'] for node_index in sorted_indices]
            self.sorted_point_clusters[key] = [self.intersection_data[e]['point'] for e in self.sorted_edge_clusters[key]]
            if 'gradient' in (self.intersection_data[(self.sorted_edge_clusters[key][0])].keys()):
                self.sorted_gradient_clusters[key] = [self.intersection_data[e]['gradient'] for e in self.sorted_edge_clusters[key]]
            if self.output_edge:
                self.sorted_t_clusters[key] = [self.intersection_data[e]['t'] for e in self.sorted_edge_clusters[key]]
     

            # self.sorted_point_gradient_clusters[key] = [self.intersection_data[e] for e in self.sorted_edge_clusters[key]]
            # print("self.sorted_point_gradient_clusters.keys()",key)
            # print(key,len( self.sorted_point_clusters[key]),len( self.sorted_gradient_clusters[key]))
            # print(self.sorted_point_gradient_clusters[(self.sorted_edge_clusters[key])[0]])
        self.label_closed_paths()
    
    def add_to_vertical_layers_manager(self, vertical_layers_manager:verticalLayerManager_dart,layer,output_edge=False):
        no_path=True
        for key in self.sorted_point_clusters:
            
            pts = self.sorted_point_clusters[key]
            if self.sorted_gradient_clusters != {}:
                grs = self.sorted_gradient_clusters[key]
            else:
                grs=None
            if self.output_edge:
                tt = self.sorted_t_clusters[key]
            if len(pts) > 3:  # discard curves that are too small
                no_path=False
                #print(pts)
                if output_edge:
                    path = path_dart(points=pts, is_closed=self.closed_paths_booleans[key],gradients=grs,edges=self.sorted_edge_clusters[key],tt=tt)
                else:
                    path = path_dart(points=pts, is_closed=self.closed_paths_booleans[key],gradients=grs)

                vertical_layers_manager.add_with_layer(path,layer-self.discard_layer_num)  
        if no_path:
            self.discard_layer_num+=1

     
        return self.edges_needed,self.discard_layer_num

    def find_intersections(self):
        """
        Fills in the
        dict self.intersection_data: key=(ui,vi) : [xi,yi,zi],
        dict self.edge_to_index: key=(u1,v1) : point_index. """
        edges_needed=self.edges_needed
        #self.edges_needed=[]
        for edge in edges_needed:
            if_intersect=self.edge_is_intersected(edge[0], edge[1])
            if if_intersect == 0:
                #self.edges_needed.append(edge)
                if self.output_edge:
                    point,t=self.find_zero_crossing_data(edge[0], edge[1])
                else:
                    point = self.find_zero_crossing_data(edge[0], edge[1])
                if self.check_gradient:
                    gradient=self.edge_gradient[edge]
                if point:  # Sometimes the result can be None
                    if edge not in self.intersection_data and tuple(reversed(edge)) not in self.intersection_data:
                        # create [edge - point] dictionary
                        self.intersection_data[edge] = {}
                        self.intersection_data[edge]['point'] = Point(point[0], point[1], point[2])
                        if self.check_gradient:
                            self.intersection_data[edge]['gradient']=gradient
                        if self.output_edge:
                            self.intersection_data[edge]['t']=t
            elif if_intersect==1:
                pass
                #self.edges_needed.append(edge)

            # create [edge - point] dictionary
            for i, e in enumerate(self.intersection_data):
                self.edge_to_index[e] = i 
    def edge_is_intersected(self, u, v):
        """ Returns True if the edge u,v has a zero-crossing,0 is intersect, 1 point is higher, 2 point is lower. """
        try:
            d1 = self.mesh.vertex[u]['scalar_field']
            d2 = self.mesh.vertex[v]['scalar_field']
            # if d1==0 or d2 ==0:
            #     print(u,v,d1,d2)
            if (d1 > 0 and d2 > 0):
                return 1
            elif (d1 < 0 and d2 < 0):
                return 2
            else:
                return 0
        except:
            #print("edge_bug",u,v)
            pass
     



    def find_zero_crossing_data(self, u, v):
        """ Finds the position of the zero-crossing on the edge u,v. """
        dist_a, dist_b = self.mesh.vertex[u]['scalar_field'], self.mesh.vertex[v]['scalar_field']
        if abs(dist_a) + abs(dist_b) > 0:
            v_coords_a, v_coords_b = self.mesh.vertex_coordinates(u), self.mesh.vertex_coordinates(v)
            
            vec = Vector.from_start_end(v_coords_a, v_coords_b)
            t= abs(dist_a) / (abs(dist_a) + abs(dist_b))
            vec = scale_vector(vec, t)
            pt = add_vectors(v_coords_a, vec)
            if self.output_edge:
                return pt,t

            return pt




# class ScalarFieldContoursweighed(ScalarFieldContours):
#     def __init__(self,mesh:Mesh,weigh):
#         ScalarFieldContours.__init__(self, mesh)  # initialize from parent class
#         self.weigh=weigh
    
#     def edge_is_intersected(self, u, v):
#         """ Returns True if the edge u,v has a zero-crossing, False otherwise. """
#         d1 = self.mesh.vertex[u]['scalar_field']
#         d2 = self.mesh.vertex[v]['scalar_field']
#         weigh=self.weigh
#         if (d1 > weigh and d2 > weigh) or (d1 < weigh and d2 < weigh):
#             return False
#         else:
#             return True   
#     def find_zero_crossing_data(self, u, v):
#         """ Finds the position of the zero-crossing on the edge u,v. """
#         weigh=self.weigh
#         dist_a, dist_b = (self.mesh.vertex[u]['scalar_field']-weigh), (self.mesh.vertex[v]['scalar_field']-weigh)
#         if abs(dist_a) + abs(dist_b) > 0:
#             v_coords_a, v_coords_b = self.mesh.vertex_coordinates(u), self.mesh.vertex_coordinates(v)
#             vec = Vector.from_start_end(v_coords_a, v_coords_b)
#             vec = scale_vector(vec, abs(dist_a) / (abs(dist_a) + abs(dist_b)))
#             pt = add_vectors(v_coords_a, vec)
#             return pt


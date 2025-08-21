import numpy as np
import trimesh
from scipy.spatial import KDTree
from compas.datastructures import Mesh
from gradient_evaluation_dart import get_face_gradient_from_scalar_field
from collections import defaultdict




class StreamlineTracer:
    def __init__(self, mesh:Mesh,  end_vertices:list,gradients=None,branching_datas = None):
        """
        Initialize the tracer with the mesh, gradient field, and stopping edges.
        :param vertices: (N, 3) array of vertex positions.
        :param faces: (M, 3) array of face indices.
        :param gradients: (M, 3) array of face gradient vectors.
        :param end_edges: Set of edges where streamlines should stop.
        """
        self.mesh=mesh
        self.load_end_vertices(end_vertices)
                

        
        #print (gradients)
        if gradients is None:
            scalar_field_list=[mesh.vertex[v]['scalar_field'] for v in mesh.vertices()]
            gradients=get_face_gradient_from_scalar_field(mesh,scalar_field_list,True)
        
        self.gradients = np.array(gradients)
        
        self.all_streamline=[]
        self.all_streamline_length={}
        self.all_vertices_length={}
        self.all_vertices_bi={}

        # Build a spatial search tree for fast nearest neighbor lookup
        #self.kdtree = KDTree(self.mesh.vertices())
        if branching_datas==None:
            self.get_branching_edge()
        else:
            self.branching_edge_datas=branching_datas
        # for face in self.branching_edge_datas:
        #     for edge in self.branching_edge_datas[face]:
        #         if self.branching_edge_datas[face][edge]!=None and self.branching_edge_datas[face][edge]>=0:
        #             print(face,edge,self.branching_edge_datas[face][edge])
    def load_end_vertices(self,end_vertices,add=0):
        self.end_vertices = {}
        self.end_edges = set()
        self.end_faces = set()
        for bi,b_vs in enumerate(end_vertices):
            for v in b_vs:
                self.end_vertices[v]=-1-bi-add
                edges = self.mesh.vertex_edges(v)
                for edge in edges:
                    if edge[0] in b_vs and edge[1] in b_vs:
                        self.end_edges.add(edge)
                        self.end_faces.update(self.mesh.edge_faces(edge[0],edge[1]))
    def start_from_vertex(self,start_vertex):
        
        all_direction_points=[]
        all_direction_edges=[]
        all_direction_faces=[]
        all_direction_points_gradients=[]
        all_direction_vkeys=[]
        all_direction_vkeys_gradients=[]
        for face in self.mesh.vertex_faces(start_vertex):
            face_other_v = [fv for fv in self.mesh.face_vertices(face) if fv != start_vertex ]
            if self.branching_edge_datas[face][(start_vertex,face_other_v[0])]>=0:
                if self.branching_edge_datas[face][(start_vertex,face_other_v[1])]<0:

                    all_direction_vkeys.append( self.branching_edge_datas[face][(start_vertex,face_other_v[0])])
                    direction_vector = np.array(self.mesh.vertex_coordinates(face_other_v[0]))-np.array(self.mesh.vertex_coordinates(start_vertex))
                    direction_vector /= np.linalg.norm(direction_vector)
                    all_direction_vkeys_gradients.append(np.dot(self.gradients[face],direction_vector))
            elif self.branching_edge_datas[face][(start_vertex,face_other_v[1])]>=0:
                all_direction_vkeys.append( self.branching_edge_datas[face][(start_vertex,face_other_v[1])])
                direction_vector = np.array(self.mesh.vertex_coordinates(face_other_v[1]))-np.array(self.mesh.vertex_coordinates(start_vertex))
                direction_vector /= np.linalg.norm(direction_vector)
                all_direction_vkeys_gradients.append(np.dot(self.gradients[face],direction_vector))
            elif self.branching_edge_datas[face][(start_vertex,face_other_v[0])]==-2 or self.branching_edge_datas[face][(start_vertex,face_other_v[1])] == -2:
                continue
                #print(face_other_v)
            else:
                pt = self.get_intersection_point_gra_vkeys(self.mesh.vertex_coordinates(start_vertex),self.gradients[face],face_other_v)
                #print(pt,face,face_other_v,self.branching_edge_datas[face][(start_vertex,face_other_v[0])],self.branching_edge_datas[face][(start_vertex,face_other_v[1])],self.gradients[face])
                if pt is not None:
                    all_direction_points.append( pt)
                    all_direction_points_gradients.append(np.linalg.norm(self.gradients[face]))
                    all_direction_faces.append(face)
                    all_direction_edges.append(tuple(face_other_v))
        if all_direction_points:
            next_point,npdn,next_edge,face = max(zip(all_direction_points,all_direction_points_gradients,all_direction_edges,all_direction_faces),key=lambda x:x[1])

        else:
            npdn =-1
        if all_direction_vkeys:
            next_vkey,nvdn = max(zip(all_direction_vkeys,all_direction_vkeys_gradients),key=lambda x:x[1])
        else:
            nvdn=-1
        
        if npdn==-1 and nvdn==-1:
            print(start_vertex)
            for face in self.mesh.vertex_faces(start_vertex):
                face_other_v = [fv for fv in self.mesh.face_vertices(face) if fv != start_vertex ]
                print(self.branching_edge_datas[face][(start_vertex,face_other_v[0])])
                print(self.branching_edge_datas[face][(start_vertex,face_other_v[1])])
                print(self.gradients[face])

            raise ValueError('start vertex islocal maximun ofr minimun')
        #print(all_direction_points)
        #print(all_direction_vkeys)
        #print(all_direction_points_gradients)
        #print(all_direction_vkeys_gradients)
        if npdn>nvdn:
            if face in self.end_faces:
                if next_edge[0] in self.end_vertices and next_edge[1] in self.end_vertices:
                    return False,self.end_vertices[next_edge[0]],None,None,np.linalg.norm(next_point-np.array(self.mesh.vertex_coordinates(start_vertex)))
            if self.branching_edge_datas[face][next_edge]>=0:
                return False,self.branching_edge_datas[face][next_edge],None,None,np.linalg.norm(np.array(self.mesh.vertex_coordinates(self.branching_edge_datas[face][next_edge]))-np.array(self.mesh.vertex_coordinates(start_vertex)))
            opposit_face=[f for f in self.mesh.edge_faces(next_edge[0],next_edge[1]) if f!=face][0]
            #print(face,opposit_face,self.branching_edge_datas[face][next_edge],self.branching_edge_datas[opposit_face][next_edge])
            return True,next_point,next_edge,opposit_face,np.linalg.norm(next_point-np.array(self.mesh.vertex_coordinates(start_vertex)))
        else:
            return False,next_vkey,None,None,np.linalg.norm(np.array(self.mesh.vertex_coordinates(next_vkey))-np.array(self.mesh.vertex_coordinates(start_vertex)))

    def trace_streamline(self, start_vertex, max_steps=10000):
        """
        Trace a streamline from a given vertex.
        :param start_vertex: Index of the starting vertex.
        :param step_size: Step size for integration.
        :param max_steps: Maximum number of steps.
        :return: Length of the streamline.
        """
        # if the start vertex is in end vertices
        if start_vertex in self.end_vertices:
            self.all_vertices_bi[start_vertex] = self.end_vertices[start_vertex]
            self.all_vertices_length[start_vertex]=0
            self.all_streamline.append((start_vertex,self.end_vertices[start_vertex]))
            self.all_streamline_length[(start_vertex,self.end_vertices[start_vertex])]=0
            return 0,self.all_vertices_bi[start_vertex]
        
        # get the next point
        continuing,next_pv,next_edge,next_face,distance=self.start_from_vertex(start_vertex)
        
        # with next point continous searching next point until touch vertex or end edges
        if continuing:

            next_pv,distance=self.trace_streamline_from_point(next_pv,next_edge,next_face,distance)
        if next_pv==start_vertex:
            raise ValueError('start vertex islocal maximun or minimun')
     
        # creat stream line between vertex to (vertex or boundary)
        self.all_streamline.append((start_vertex,next_pv))
        self.all_streamline_length[(start_vertex,next_pv)]=(distance)  
        # if next point is end vertex, return distance and the item of boundary
        if next_pv<0:
            print('touch boundary',self.all_streamline[-1])
            self.all_vertices_length[start_vertex]=distance
            self.all_vertices_bi[start_vertex]=next_pv
            return distance,next_pv
              
        # if next point is not end vertex, continous tracing to next vertex or boundary
        else:
            print('touch vertex',self.all_streamline[-1])
            if next_pv in self.all_vertices_length:
                distance_add,relative_b = self.all_vertices_length[next_pv],self.all_vertices_bi[next_pv]
            else:
                distance_add,relative_b = self.trace_streamline(next_pv)

            # give start vertex the whole distance from it to the boundary
            self.all_vertices_length[start_vertex]=distance+distance_add
            self.all_vertices_bi[start_vertex]=relative_b
            
            return self.all_vertices_length[start_vertex],relative_b

        for _ in range(max_steps):
            # Find the nearest vertex
            nearest_index = self.kdtree.query(pos)[1]

            # Check if we reached a stopping edge
            for neighbor in self.faces[np.any(self.faces == nearest_index, axis=1)]:
                for edge in [(neighbor[i], neighbor[(i + 1) % 3]) for i in range(3)]:
                    if tuple(sorted(edge)) in self.end_edges:
                        return total_length  # Stop if we hit an end edge

            # Get gradient direction
            grad = self.vertex_gradients[nearest_index]


            # Move along gradient
            direction = grad / np.linalg.norm(grad)
            new_pos = pos + step_size * direction

            # Update length and position
            total_length += np.linalg.norm(new_pos - pos)
            pos = new_pos

        return total_length
    def trace_streamline_from_point(self,point,edge,face,distance=0):
        """
        return -(x+1),distance if touch boundary 
                x,distance if touch vertex
        """
        end_check = False
        
        if face in self.end_faces:
            if edge[0] in self.end_vertices and edge[1] in self.end_vertices:
                
                return self.end_vertices[edge[0]],distance
            else:
                end_check = True
        
        # try:
        #     print('edge',self.branching_edge_datas[face][edge])
        # except:
        #     print('face',face,self.mesh.face_vertices(face),edge)
        # if self.branching_edge_datas[face][edge]>=0:
        #     return self.branching_edge_datas[face][edge],distance+np.linalg.norm(np.array(self.mesh.vertex_coordinates(self.branching_edge_datas[face][edge]))-(point))
        #print('trace from point',point,edge,face,self.mesh.face_vertices(face))
        fvs = self.mesh.face_vertices(face)
        edges_to_check=[(fvs[i],fvs[i-1]) for i in range(3) if fvs[i] not in edge or fvs[i-1] not in edge ]
        #print('edges to check',edges_to_check)
        for edge_to_check in edges_to_check:
            #print ('edge to check',self.branching_edge_datas[face][edge_to_check])
            if self.branching_edge_datas[face][edge_to_check] == -1 :
                # edge to check can be going in from outside meaning it can not be going to from in side with a same directino
                continue
                return self.branching_edge_datas[face][edge_to_check],distance+np.linalg.norm(np.array(self.mesh.vertex_coordinates(self.branching_edge_datas[face][edge]))-(point))
            else:
                next_point = self.get_intersection_point_gra_vkeys(point,self.gradients[face],edge_to_check)
                #print(next_point)
                if next_point is None:

                    #raise Exception('branching data not good enough for edge')
                    continue
                if end_check:
                    if edge_to_check[0] in self.end_vertices and edge_to_check[1] in self.end_vertices:
                        distance+=np.linalg.norm(next_point-point)
                        return self.end_vertices[edge_to_check[0]],distance

                if self.branching_edge_datas[face][edge_to_check]>=0:
                    return self.branching_edge_datas[face][edge_to_check],distance+np.linalg.norm(np.array(self.mesh.vertex_coordinates(self.branching_edge_datas[face][edge_to_check]))-(point))
                distance+=np.linalg.norm(next_point-point)
                opposite_face = [x for x in self.mesh.edge_faces(edge_to_check[0],edge_to_check[1]) if x != face][0]
             
                return self.trace_streamline_from_point(next_point,edge_to_check,opposite_face,distance)   
        #print([self.mesh.vertex_coordinates(x) for x in edge])
        for face_i in self.mesh.edge_faces(edge[0],edge[1]):
            print(self.branching_edge_datas[face_i][edge])
      
        for edge_to_check in edges_to_check:
            print('start point',point,'checked edge',self.mesh.vertex_coordinates(edge_to_check[0]),self.mesh.vertex_coordinates(edge_to_check[1]))
            print('gradiengts',self.gradients[face])
            print(self.get_intersection_point_gra_vkeys(point,self.gradients[face],edge_to_check))
            edge_vector=(np.array(self.mesh.vertex_coordinates(edge[0]))-np.array(self.mesh.vertex_coordinates(edge[1])))
            opposit_vertex = [x for x in self.mesh.face_vertices(face) if x not in edge][0]
            point_vector=-(point-np.array(self.mesh.vertex_coordinates(opposit_vertex)))
            edge1_vector=-np.array(self.mesh.vertex_coordinates(edge[0]))+np.array(self.mesh.vertex_coordinates(opposit_vertex))
            edge2_vector=-np.array(self.mesh.vertex_coordinates(edge[1]))+np.array(self.mesh.vertex_coordinates(opposit_vertex))
            print(are_on_same_side(point_vector,edge_vector,self.gradients[face]))
            print(are_on_same_side(edge1_vector,edge_vector,self.gradients[face]))
            print(are_on_same_side(edge2_vector,edge_vector,self.gradients[face]))
            

        raise ValueError("No edge found")
    def compute_streamline_lengths(self):
        """
        Compute streamline lengths for all vertices.
        :return: Array of streamline lengths.
        """
      
        for v in (self.mesh.vertices()):
            if v not in self.all_vertices_length:
                self.trace_streamline(v)
        return self.all_vertices_length

    def get_branching_edge(self):
        print("get branching edge")
        self.branching_edge_datas={}
        for face in self.mesh.faces():
            self.branching_edge_datas[face]={}
            face_vertices_key = self.mesh.face_vertices(face)
            face_vertices = np.array([self.mesh.vertex_coordinates(v) for v in face_vertices_key])
            face_gradient = self.gradients[face]
            for i in range(3):
                A = face_vertices[(i+1)%3]-face_vertices[i]
                B = face_vertices[i-1]-face_vertices[i]
                
                going_in = are_on_same_side(A,B,face_gradient)
                #print((face_vertices[i],face_vertices[i-1]),type((face_vertices[i],face_vertices[i-1])))
                if going_in:
                    
                    self.branching_edge_datas[face][(face_vertices_key[i],face_vertices_key[i-1])]=-1
                    self.branching_edge_datas[face][(face_vertices_key[i-1],face_vertices_key[i])]=-1
                else:
                    self.branching_edge_datas[face][(face_vertices_key[i],face_vertices_key[i-1])]=None
                    self.branching_edge_datas[face][(face_vertices_key[i-1],face_vertices_key[i])]=None
        for face in self.branching_edge_datas:
       
            for edge in self.branching_edge_datas[face]:
                if edge[0]>edge[1]:
                    continue
              
                if self.branching_edge_datas[face][edge]==None:
                    opposite_face = [x for x in self.mesh.edge_faces(edge[0],edge[1]) if x != face][0]
                    #print(opposite_face)
                    if opposite_face is None or self.branching_edge_datas[opposite_face][edge] is None or self.branching_edge_datas[opposite_face][edge] >=0  :
                        # if opposite_face is not None:
                        #     print('branch',edge,face,opposite_face)
                        A = np.array(self.mesh.vertex_coordinates(edge[0]))-np.array(self.mesh.vertex_coordinates(edge[1]))
                        if np.dot(A,self.gradients[face])>0:
                            self.branching_edge_datas[face][edge]=edge[0]
                            self.branching_edge_datas[face][(edge[1],edge[0])]=edge[0]
                        else:
                            self.branching_edge_datas[face][edge]=edge[1]
                            self.branching_edge_datas[face][(edge[1],edge[0])]=edge[1]
                        # if opposite_face is not None and self.branching_edge_datas[opposite_face][edge] is not None and self.branching_edge_datas[face][edge] != self.branching_edge_datas[opposite_face][edge]:
                        #     print('strange situation',face,opposite_face,edge)
                    # if 2150 in (face,opposite_face) and 9753 in (face,opposite_face):
                    #     print(face,opposite_face,edge)
                    #     print(self.branching_edge_datas[face][edge])
                    #     print(self.branching_edge_datas[face][(edge[1],edge[0])])
                    #     print(self.branching_edge_datas[opposite_face][(edge[1],edge[0])])
                    #     print(self.branching_edge_datas[opposite_face][edge])
        for face in self.branching_edge_datas:
            for edge in self.branching_edge_datas[face]:
                if self.branching_edge_datas[face][edge]==None:
                    self.branching_edge_datas[face][edge]=-2
    def get_intersection_point_gra_vkeys(self,point,gradient,vkeys):
        
        B = (self.mesh.vertex_coordinates(vkeys[0]))
        C = (self.mesh.vertex_coordinates(vkeys[1]))

        return find_intersection_gra([point,B,C],gradient) 

class StreamlineTracer_double_side(StreamlineTracer):

    def __init__(self, mesh:Mesh, end_vertices_High, end_vertices_Low,gradients=None,branching_datas=None):
        self.end_vertices_High=end_vertices_High
        self.end_vertices_Low=end_vertices_Low
        super().__init__(mesh,self.end_vertices_High,gradients=gradients,branching_datas=branching_datas)


    def trace_streamlines_both_side(self):
        self.compute_streamline_lengths()

        self.vertices_length_uping=self.all_vertices_length
        self.vertices_bi_uping=self.all_vertices_bi
        self.all_vertices_length={}
        self.all_vertices_bi={}
        
        self.flip_streamline_length()

        self.gradients=-self.gradients
        self.get_branching_edge()
        self.load_end_vertices(self.end_vertices_Low,len(self.end_vertices_High))
        self.compute_streamline_lengths()

        self.vertices_length_downing=self.all_vertices_length
        self.vertices_bi_downing=self.all_vertices_bi
    def flip_streamline_length(self):
        self.all_streamline = [ (x[1],x[0]) for x in self.all_streamline]
        self.all_streamline_length = {(x[1],x[0]): self.all_streamline_length[x] for x in self.all_streamline_length}
    
    def creat_compound_target_length_simple(self,preprocessor):
        # A=(list(self.vertices_length_uping.keys()))
        # print (A)
        # A.sort()
        # print(A)
        # print(len(A))
        # vertices = list(self.mesh.vertices())
        # print(len(vertices))
        # print([x for x in vertices if x  not in A])
        preprocessor.target_HIGH.offset_distance_list = [self.vertices_length_uping[v] for v in self.mesh.vertices()]
        preprocessor.target_LOW.offset_distance_list = [self.vertices_length_downing[v] for v in self.mesh.vertices()]
        print(np.array(preprocessor.target_HIGH.offset_distance_list)+np.array(preprocessor.target_LOW.offset_distance_list))

    def save(self,Outputpath):
        
        s={
            "streamline":self.all_streamline,
            "streamlinelength":[self.all_streamline_length[x] for x in self.all_streamline],


            "lengthuping":self.vertices_length_uping,
            "lengthdowning":self.vertices_length_downing,

            "biuping":self.vertices_bi_uping,
            "bidowning":self.vertices_bi_downing,

          
            "gradients":self.gradients.tolist(),
            "end_vertices_High":self.end_vertices_High,
            "end_vertices_Low":self.end_vertices_Low,

         

        }
        
        import compas_slicer.utilities as utils
        utils.save_to_json(data=s,filepath=Outputpath,name="streamline_double_side.json")
    
    @classmethod
    def load(cls,Inputpath,mesh):
        import compas_slicer.utilities as utils
        print('load streamline double side')
        s=utils.load_from_json(filepath=Inputpath,name="streamline_double_side.json")
        print('set data')
        gradients = np.array(s["gradients"])
        #print(gradients)
        object_i =  cls(mesh,s["end_vertices_High"],s["end_vertices_Low"],gradients=gradients )
        object_i.all_streamline=s["streamline"]
        object_i.all_streamline_length={tuple(k):v for k,v in zip(s["streamline"],s["streamlinelength"])}

        object_i.vertices_length_uping={int(key):s["lengthuping"][key] for key in s["lengthuping"]}
        object_i.vertices_length_downing={int(key):s["lengthdowning"][key] for key in s["lengthdowning"]}

        object_i.vertices_bi_uping={int(key):s["biuping"][key] for key in s["biuping"]}
        object_i.vertices_bi_downing={int(key):s["bidowning"][key] for key in s["bidowning"]}

        return object_i
    
class streamline_topology(StreamlineTracer_double_side):
    def __init__(self, mesh:Mesh, end_vertices_High, end_vertices_Low,gradients=None):
        super().__init__(mesh, end_vertices_High, end_vertices_Low,gradients=gradients)
        self.all_streamline_topology={}
        self.all_streamline_topology[0]=[]
        self.all_streamline_topology[1]=[]
    @classmethod
    def load(cls, Inputpath, mesh):
        return super().load(Inputpath, mesh)   
    def get_topo(self):
        new_graph = apply_merging(self.all_streamline_length)
        print("\n合并后的图:")
        print(new_graph)
    
    def get_max_distances(self):
        start_nodes = list(range(-len(self.end_vertices_Low),0))
        end_nodes = list(range((-len(self.end_vertices_High))-len(self.end_vertices_Low),-len(self.end_vertices_Low)))
        from_h=(compute_node_values_multisource(self.all_streamline_length,start_nodes))
        self.flip_streamline_length()
        from_l=(compute_node_values_multisource(self.all_streamline_length,end_nodes))
        self.flip_streamline_length()
        keys = list(from_l.keys())
        missed_keys = [v for v in self.mesh.vertices() if v not in keys]
        #print(missed_keys)
        print([from_l[v]-self.vertices_length_downing[v] for v in self.mesh.vertices()])
        print([from_h[v]-self.vertices_length_uping[v] for v in self.mesh.vertices()])
        self.vertices_length_downing = from_l
        #self.vertices_length_uping = from_h
def apply_merging(graph):
    """应用合并操作到图中所有可能的路径对"""
    # 找出所有节点对(u,v)及其所有路径
    print('apply_merging')
    nodes = set()
    for u, v in graph:
        nodes.add(u)
        nodes.add(v)
    
    path_dict = defaultdict(list)
    for u in nodes:
        for v in nodes:
            if u != v:
                paths = find_all_paths(graph, u, v)
                if len(paths) >= 2:
                    path_dict[(u, v)] = paths
    
    # 合并所有可能的路径对
    new_graph = graph.copy()
    for (u, v), paths in path_dict.items():
        for i in range(len(paths)):
            for j in range(i+1, len(paths)):
                merged_edges = merge_paths(graph, paths[i], paths[j])
                new_graph.update(merged_edges)
    
    return new_graph
def find_all_paths(graph, start, end, path=None):
    """查找图中从start到end的所有路径"""
    #print('find_all_paths')
    if path is None:
        path = []
    path = path + [start]
    if start == end:
        return [path]
    paths = []
    for (u, v) in graph:
        if u == start and v not in path:  # 避免循环
            new_paths = find_all_paths(graph, v, end, path)
            for p in new_paths:
                print(p)
                paths.append(p)
    return paths
def merge_paths(graph, path1, path2):
    """
    合并两条路径，生成新的路径并重新分配边长度。
    
    参数:
        graph: 原始图的边字典，格式为 {(u, v): length}
        path1: 第一条路径，顶点列表，如 [A, B, C]
        path2: 第二条路径，顶点列表，如 [A, D, C]
    
    返回:
        新的图的边字典，包含合并后的路径。
    """
    # 检查两条路径的起点和终点是否相同
    if path1[0] != path2[0] or path1[-1] != path2[-1]:
        raise ValueError("Paths must have the same start and end nodes.")
    
    # 计算两条路径的总长度
    def path_length(path):
        length = 0
        for i in range(len(path) - 1):
            length += graph.get((path[i], path[i+1]), 0)
        return length
    
    L1 = path_length(path1)
    L2 = path_length(path2)
    L_new = max(L1, L2)
    
    # 构造新路径的顶点序列（交替合并两条路径的中间顶点）
    new_path = [path1[0]]  # 起点
    i, j = 1, 1  # 跳过起点
    while i < len(path1) - 1 or j < len(path2) - 1:
        if i < len(path1) - 1:
            new_path.append(path1[i])
            i += 1
        if j < len(path2) - 1:
            new_path.append(path2[j])
            j += 1
    new_path.append(path1[-1])  # 终点
    
    # 重新分配边长度
    new_edges = {}
    
    # 计算 path1 中顶点的累积比例
    cum_length = 0
    ratios = {}
    for i in range(len(path1)):
        ratios[path1[i]] = cum_length / L1
        if i < len(path1) - 1:
            cum_length += graph.get((path1[i], path1[i+1]), 0)
    
    # 计算新路径中各边的长度
    cum_new_length = 0
    for i in range(len(new_path) - 1):
        u, v = new_path[i], new_path[i+1]
        # 如果这条边在原 path1 中存在，则按比例分配
        if (u, v) in graph and u in ratios and v in ratios:
            ratio = ratios[v] - ratios[u]
            new_length = L_new * ratio
        else:
            # 否则保持原长度（如 path2 中的边）
            new_length = graph.get((u, v), 0)
        new_edges[(u, v)] = new_length
    
    return new_edges
def compute_node_values_multisource(graph, start_nodes):
    # 初始化
    values = {}
    for edge in graph:
        s, e = edge
        values[s] = float('-inf')
        values[e] = float('-inf')
    
    # 所有起始节点值为0
    for node in start_nodes:
        values[node] = 0
    
    # 构建邻接表
    adj = {}
    for (s, e), l in graph.items():
        if s not in adj:
            adj[s] = []
        adj[s].append((e, l))
    
    # Bellman-Ford算法变种（处理最长路径）
    nodes = list(values.keys())
    for _ in range(len(nodes) - 1):
        updated = False
        for s in adj:
            # 如果s尚未可达则跳过
            if values[s] == float('-inf'):
                continue
                
            for e, l in adj[s]:
                if values[s] + l > values[e]:
                    values[e] = values[s] + l
                    updated = True
        if not updated:
            break
    
    # 检查是否有正权环
    for s in adj:
        if values[s] == float('-inf'):
            continue
        for e, l in adj[s]:
            if values[s] + l > values[e]:
                print(f"警告：图中存在正权环，无法确定某些节点的最终值")
                break
    
    # 处理不可达节点
    for node in values:
        if values[node] == float('-inf'):
            values[node] = None
    
    return values
class StreamlineTracer_double_side_with_polylines(StreamlineTracer_double_side):

    def __init__(self, mesh:Mesh, end_vertices_High, end_vertices_Low,gradients=None):
        super().__init__(mesh, end_vertices_High, end_vertices_Low,gradients=gradients)
        self.all_streamline_polylines={}
        self.all_streamline_polylines[0]=[]
        #self.all_streamline_polylines[1]=[]
    def trace_streamline_from_point(self,point,edge,face,distance=0):
        """
        return -(x+1),distance if touch boundary 
                x,distance if touch vertex
        """
        end_check = False
        self.all_streamline_polylines[0].append(point)
        if face in self.end_faces:
            if edge[0] in self.end_vertices and edge[1] in self.end_vertices:
                
                
                return self.end_vertices[edge[0]],distance
            else:
                end_check = True
        

        fvs = self.mesh.face_vertices(face)
        edges_to_check=[(fvs[i],fvs[i-1]) for i in range(3) if fvs[i] not in edge or fvs[i-1] not in edge ]
        #print('edges to check',edges_to_check)
        for edge_to_check in edges_to_check:
            #print ('edge to check',self.branching_edge_datas[face][edge_to_check])
            if self.branching_edge_datas[face][edge_to_check] == -1 :
                # edge to check can be going in from outside meaning it can not be going to from in side with a same directino
                continue
               
            else:
                next_point = self.get_intersection_point_gra_vkeys(point,self.gradients[face],edge_to_check)
                #print(next_point)
                if next_point is None:

                    #raise Exception('branching data not good enough for edge')
                    continue
                if end_check:
                    if edge_to_check[0] in self.end_vertices and edge_to_check[1] in self.end_vertices:
                        distance+=np.linalg.norm(next_point-point)
                        return self.end_vertices[edge_to_check[0]],distance

                if self.branching_edge_datas[face][edge_to_check]>=0:
                    return self.branching_edge_datas[face][edge_to_check],distance+np.linalg.norm(np.array(self.mesh.vertex_coordinates(self.branching_edge_datas[face][edge_to_check]))-(point))
                distance+=np.linalg.norm(next_point-point)
                opposite_face = [x for x in self.mesh.edge_faces(edge_to_check[0],edge_to_check[1]) if x != face][0]

                return self.trace_streamline_from_point(next_point,edge_to_check,opposite_face,distance)   
        #print([self.mesh.vertex_coordinates(x) for x in edge])
        for face_i in self.mesh.edge_faces(edge[0],edge[1]):
            print(self.branching_edge_datas[face_i][edge])
      
        for edge_to_check in edges_to_check:
            print('start point',point,'checked edge',self.mesh.vertex_coordinates(edge_to_check[0]),self.mesh.vertex_coordinates(edge_to_check[1]))
            print('gradiengts',self.gradients[face])
            print(self.get_intersection_point_gra_vkeys(point,self.gradients[face],edge_to_check))
            edge_vector=(np.array(self.mesh.vertex_coordinates(edge[0]))-np.array(self.mesh.vertex_coordinates(edge[1])))
            opposit_vertex = [x for x in self.mesh.face_vertices(face) if x not in edge][0]
            point_vector=-(point-np.array(self.mesh.vertex_coordinates(opposit_vertex)))
            edge1_vector=-np.array(self.mesh.vertex_coordinates(edge[0]))+np.array(self.mesh.vertex_coordinates(opposit_vertex))
            edge2_vector=-np.array(self.mesh.vertex_coordinates(edge[1]))+np.array(self.mesh.vertex_coordinates(opposit_vertex))
            print(are_on_same_side(point_vector,edge_vector,self.gradients[face]))
            print(are_on_same_side(edge1_vector,edge_vector,self.gradients[face]))
            print(are_on_same_side(edge2_vector,edge_vector,self.gradients[face]))
            

        raise ValueError("No edge found")    
    
    def trace_streamline(self, start_vertex, max_steps=10000):
        """
        Trace a streamline from a given vertex.
        :param start_vertex: Index of the starting vertex.
        :param step_size: Step size for integration.
        :param max_steps: Maximum number of steps.
        :return: Length of the streamline.
        """
        # if the start vertex is in end vertices
        self.all_streamline_polylines[0].append(self.mesh.vertex_coordinates(start_vertex))
        if start_vertex in self.end_vertices:
            self.all_vertices_bi[start_vertex] = self.end_vertices[start_vertex]
            self.all_vertices_length[start_vertex]=0
            self.all_streamline.append((start_vertex,self.end_vertices[start_vertex]))

            self.all_streamline_length[(start_vertex,self.end_vertices[start_vertex])]=0
            
            self.all_streamline_polylines[0]=[]
            return 0,self.all_vertices_bi[start_vertex]
        
        # get the next point
        continuing,next_pv,next_edge,next_face,distance=self.start_from_vertex(start_vertex)

        # with next point continous searching next point until touch vertex or end edges
        if continuing:

            next_pv,distance=self.trace_streamline_from_point(next_pv,next_edge,next_face,distance)
        
        # creat stream line between vertex to (vertex or boundary)
        self.all_streamline.append((start_vertex,next_pv))
        self.all_streamline_length[(start_vertex,next_pv)]=(distance)  
        poly_line=self.all_streamline_polylines[0]
        if next_pv >=0:
            self.all_streamline_polylines[0].append(self.mesh.vertex_coordinates(next_pv))
        self.all_streamline_polylines[(start_vertex,next_pv)]=poly_line
  
        #print(self.all_streamline_polylines)
        
        
        self.all_streamline_polylines[0]=[]
        # if next point is end vertex, return distance and the item of boundary
        if next_pv == start_vertex:
            print(start_vertex,continuing)
            raise ValueError("Loop")
        if next_pv<0:
            #print('touch boundary',self.all_streamline[-1])
            self.all_vertices_length[start_vertex]=distance
            self.all_vertices_bi[start_vertex]=next_pv
            return distance,next_pv
              
        # if next point is not end vertex, continous tracing to next vertex or boundary
        else:
            #print('touch vertex',self.all_streamline[-1])
            if next_pv in self.all_vertices_length:
                distance_add,relative_b = self.all_vertices_length[next_pv],self.all_vertices_bi[next_pv]
            else:
                distance_add,relative_b = self.trace_streamline(next_pv)

            # give start vertex the whole distance from it to the boundary
            self.all_vertices_length[start_vertex]=distance+distance_add
            self.all_vertices_bi[start_vertex]=relative_b
            
            return self.all_vertices_length[start_vertex],relative_b

    def save_polyline(self,Output_path):
        keys = list(self.all_streamline_polylines.keys())
        paths_list=[]
        key_list=[]
        for path in keys:
            if not self.all_streamline_polylines[path]:
                del self.all_streamline_polylines[path]
            else:
                for p_i,point in enumerate(self.all_streamline_polylines[path]):
                    if not isinstance(point,list):
                        self.all_streamline_polylines[path][p_i]=point.tolist()
                paths_list.append(self.all_streamline_polylines[path])
                key_list.append(path)
        import compas_slicer.utilities as utils
        utils.save_to_json(data=paths_list, filepath=Output_path, name='streamlinepolylines.json')
        #utils.save_to_json(data=key_list, filepath=Output_path, name='streamline_polylines_keys')
def find_intersection_gra(points,gradient):
    """
    points: list of 3 points
    
    return: point

  
    
    """
 

    # 将点转换为numpy数组
    A = np.array(points[0])
    B = np.array(points[1])
    C = np.array(points[2])

    

    
    
   
    n_x, n_y, n_z = gradient

    
    # 计算t

    numerator = (n_x*(B[1]-A[1])+n_y*(A[0]-B[0]))
    denominator = -(n_x*(C[1]-B[1])+n_y*(B[0]-C[0]))
    
    if denominator == 0:
        print("分母为0法计算t")
        return None  # 没有交点
    
    t = numerator / denominator
    #print(t,'__________________')
    
    # 计算交点坐标
    if t <0:
        return None
      
    elif t>1:
        return None
    
    else:
        D = B + t * np.array(C-B)

    

    return D

def is_in_angle(A, B, C):
    def cross(u, v):
        return u[0]*v[1] - u[1]*v[0]
    
    def dot(u, v):
        return u[0]*v[0] + u[1]*v[1]
    
    def length_sq(u):
        return u[0]**2 + u[1]**2
    
    cross_AB = cross(A, B)
    cross_AC = cross(A, C)
    cross_CB = cross(C, B)
    
    if cross_AB > 0:
        return cross_AC >= 0 and cross_CB >= 0
    elif cross_AB < 0:
        return cross_AC <= 0 and cross_CB <= 0
    else:  # A and B are collinear
        dot_AB = dot(A, B)
        if dot_AB > 0:  # same direction
            dot_AC = dot(A, C)
            return dot_AC > 0 and length_sq(C) > length_sq(A)
        else:  # opposite direction
            if cross_AC != 0 or cross_CB != 0:
                return True
            else:
                dot_AC = dot(A, C)
                dot_BC = dot(B, C)
                return (dot_AC > 0 and length_sq(C) > length_sq(A)) or \
                       (dot_BC > 0 and length_sq(C) > length_sq(B))
def are_on_same_side(A, B, C):
    # 验证共面性
    if not np.isclose(np.dot(np.cross(A, B), C), 0):
        raise ValueError("Vectors are not coplanar")
    
    # 计算平面法向量
    N = np.cross(A, B)
    if np.allclose(N, [0,0,0]):  # A和B平行
        N = np.cross(A, C)  # 尝试其他组合
    
    # 构造垂直于B的平面内向量
    B_perp = np.cross(N, B)
    
    # 计算投影
    proj_A = np.dot(A, B_perp)
    proj_C = np.dot(C, B_perp)
    
    return proj_A * proj_C > 0

def main():
    import time
    import os
    from compas.datastructures import Mesh
    import logging
    import compas_slicer.utilities as utils
    from compas_slicer.slicers import InterpolationSlicer
    from compas_slicer.pre_processing import create_mesh_boundary_attributes
    from mesh_changing import change_mesh_shape
    from compas.files import OBJWriter
    from interpolationdart import DartPreprocesssor
    from interpolation_slicer_dart import InterpolationSlicer_Dart
    logger = logging.getLogger('logger')
    logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)


    input_folder_name='MNNaCl1'#'Jul_ai''whole''beam1B''csch2''example_jun_bg''data_Y_shape' 'data_vase''data_costa_surface''data_Y_shape_o''data_vase_o''data_costa_surface_o''Jun_ab_testmultipipe'
    #'Jun_ah_testb''Jul_h''Jul_I''Jul_ab''Jul_ah''Jul_ba''table_1''Aug_ac_ex''Aug_bg''Aug_bh''example_jun_bg''table_2'
    DATA_PATH = os.path.join(os.path.dirname(__file__), input_folder_name)
    OUTPUT_PATH = utils.get_output_directory(DATA_PATH)
    OBJ_INPUT_NAME = os.path.join(DATA_PATH, 'mesh.obj')
    start_time = time.time()
    avg_layer_height = 15
    try:
        
        mesh = Mesh.from_obj(os.path.join(OUTPUT_PATH,"edited_mesh.obj"))  
        #mesh = Mesh.from_obj(os.path.join(DATA_PATH, OBJ_INPUT_NAME))
        print("old mesh")
        # --- Load targets (boundaries)
        low_boundary_vs = utils.load_from_json(DATA_PATH, 'boundaryLOW.json')
        high_boundary_vs = utils.load_from_json(DATA_PATH, 'boundaryHIGH.json')
        create_mesh_boundary_attributes(mesh, low_boundary_vs, high_boundary_vs)  
    except:
        print("new mesh")
        # --- Load initial_mesh
        mesh = Mesh.from_obj(os.path.join(DATA_PATH, OBJ_INPUT_NAME))




        # --- Load targets (boundaries)
        low_boundary_vs = utils.load_from_json(DATA_PATH, 'boundaryLOW.json')
        high_boundary_vs = utils.load_from_json(DATA_PATH, 'boundaryHIGH.json')
        
        create_mesh_boundary_attributes(mesh, low_boundary_vs, high_boundary_vs)

        # print(mesh.vertex_attribute(key=6615, name="z"))
        # This part is add by Yichuan

        change_mesh_shape(mesh,-1)

        # print(mesh.vertex_attribute(key=6615, name="z"))
        obj_writer = OBJWriter(filepath= os.path.join(OUTPUT_PATH, "edited_mesh.obj"), meshes=[mesh])
        obj_writer.write()
        mesh= Mesh.from_obj(os.path.join(OUTPUT_PATH,"edited_mesh.obj"))  
        low_boundary_vs = utils.load_from_json(DATA_PATH, 'boundaryLOW.json')
        high_boundary_vs = utils.load_from_json(DATA_PATH, 'boundaryHIGH.json')
        create_mesh_boundary_attributes(mesh, low_boundary_vs, high_boundary_vs) 
        # end
    parameters = {
        'avg_layer_height': avg_layer_height,  # controls number of curves that will be generated
    }
    # for face in mesh.faces():
    #     vertices=[l_corner[1] for l_corner in mesh.face_corners(fkey=face)]
    #     ver_coor=[mesh.vertex_coordinates(vertex) for vertex in vertices]
    #     line=Line(ver_coor[0],ver_coor[1])
    #     print(face,mesh.face_corners(fkey=face)[0],line.start )

    """
    preprocessor = DartPreprocesssor(mesh, parameters, DATA_PATH,False)
    preprocessor.d_create_compound_targets()
    g_eval = preprocessor.create_gradient_evaluation(norm_filename='gradient_norm.json', g_filename='gradient.json',
                                                     target_1=preprocessor.target_LOW,
                                                     target_2=preprocessor.target_HIGH,way='final')   
    
    scalar_field=[]
    for vertex,data in mesh.vertices(data=True):
        scalar_field.append(data['scalar_field'])
    print_list_with_details(scalar_field,"scalar field")
    save_nested_list(file_path=OUTPUT_PATH,file_name="scalar_field_org.josn",nested_list=scalar_field)
    """
    preprocessor = DartPreprocesssor(mesh, parameters, DATA_PATH)
    preprocessor.d_create_compound_targets()
    
    #print_list_with_details(preprocessor.target_HIGH.offset_distance_list_for_target,"preprocessor.target_HIGH.offset_distance_list_for_target")
    

    # 求saddle points
    g_eval_height = preprocessor.create_gradient_evaluation(norm_filename='gradient_norm.json', g_filename='gradient.json',
                                                     target_1=preprocessor.target_LOW,
                                                     target_2=preprocessor.target_HIGH,way='z',show_graph=False)
 
    #saddles=preprocessor.find_critical_points_with_related_boundary(g_eval_height, output_filename= 'height_saddles.json',sort_by_height=True)
    try:
        Tracer = streamline_topology.load(OUTPUT_PATH,mesh)
    except:
       
        for _ in range(1) :   
            g_eval_height.find_critical_points()
            g_eval_height.kill_max_min_muliti_way()
            g_eval_height.find_critical_points()
            if g_eval_height.maxima or g_eval_height.minima:
                raise Exception("extreme points found")
            # Tracer=StreamlineTracer(mesh=mesh,end_vertices=preprocessor.target_HIGH.clustered_vkeys)
            # Tracer.compute_streamline_lengths()

            Tracer = streamline_topology(mesh=mesh,end_vertices_High=preprocessor.target_HIGH.clustered_vkeys,end_vertices_Low=preprocessor.target_LOW.clustered_vkeys)
            Tracer.trace_streamlines_both_side()
            Tracer.save(OUTPUT_PATH)
    Tracer.get_max_distances()
    Tracer.creat_compound_target_length_simple(preprocessor)

    g = preprocessor.create_gradient_evaluation(norm_filename='gradient_norm.json', g_filename='gradient.json',
                                                            target_1=preprocessor.target_LOW,
                                                            target_2=preprocessor.target_HIGH,way='down',show_graph=False,target_index=-1)
    
        
   
    x=70

    preprocessor.save_scalar_field()
    slicer = InterpolationSlicer_Dart(mesh, preprocessor, parameters,False,)#gradient_evaluation=G
    slicer.slice_model(weights_list=[(i+1) / x-0.001 for i in range(x)])  # compute_norm_of_gradient contours
    slicer.printout_info()
    utils.save_to_json(slicer.to_data(), OUTPUT_PATH, 'curved_slicer'+'.json')

if __name__ == "__main__":
    main()
    
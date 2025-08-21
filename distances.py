import heapq
from compas.datastructures import Mesh
from compas.geometry import Line, intersection_line_triangle, intersection_ray_mesh
import numpy as np
import time
import copy
#import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import json
sys.setrecursionlimit(10000) 

def dijkstra_distances(mesh: Mesh, source):
    # 初始化距离字典，所有顶点的距离初始为无穷大
    distances = {vertex: float("inf") for vertex in mesh.vertices()}
    distances[source] = 0

    # 优先队列，存储 (距离, 顶点) 对
    priority_queue = [(0, source)]

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        # 如果从优先队列中取出的顶点距离大于已知最短距离，则跳过
        if current_distance > distances[current_vertex]:
            continue

        for neighbor in mesh.vertex_neighbors(current_vertex):
            distance = current_distance + mesh.edge_length(
                current_vertex, neighbor
            ) * mesh.edge_attribute(
                edge=(current_vertex, neighbor), name="weigh_cur"
            )  # 假设每条边的权重为1

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances


def multi_source_dijkstra(mesh: Mesh, sources):
    print("start_multi_source_dijkstra")
    # 初始化距离字典，所有顶点的距离初始为无穷大
    distances = {vertex: float("inf") for vertex in mesh.vertices()}

    # 优先队列，存储 (距离, 顶点) 对
    priority_queue = []

    # 对于每个源点，将其距离设为0，并加入优先队列
    for source in sources:
        distances[source] = 0
        heapq.heappush(priority_queue, (0, source))

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        # 如果从优先队列中取出的顶点距离大于已知最短距离，则跳过
        if current_distance > distances[current_vertex]:
            continue
        

        for neighbor in mesh.vertex_neighbors(current_vertex):
            weigh = mesh.edge_length(current_vertex, neighbor)
            # weigh=mesh.edge_length(current_vertex,neighbor)
            # print(weigh)
            distance = current_distance + weigh  

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    distance_list = list(distances.values())
    #print("multi_source_dijkstra", distance_list)
    return distance_list


def multi_source_dijkstra_with_path(mesh: Mesh, sources):
    print("start_multi_source_dijkstra")
    # 初始化距离字典，所有顶点的距离初始为无穷大
    distances = {vertex: float("inf") for vertex in mesh.vertices()}
    # 初始化前驱字典，用于记录最短路径
    previous_vertices = {vertex: None for vertex in mesh.vertices()}

    # 优先队列，存储 (距离, 顶点) 对
    priority_queue = []

    # 对于每个源点，将其距离设为0，并加入优先队列
    for source in sources:
        distances[source] = 0
        heapq.heappush(priority_queue, (0, source))

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        # 如果从优先队列中取出的顶点距离大于已知最短距离，则跳过
        if current_distance > distances[current_vertex]:
            continue

        for neighbor in mesh.vertex_neighbors(current_vertex):
            weigh = mesh.edge_length(current_vertex, neighbor)
            distance = current_distance + weigh

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_vertices[neighbor] = current_vertex
                heapq.heappush(priority_queue, (distance, neighbor))

    # 构建从源点到所有顶点的最短路径
    paths = {vertex: [] for vertex in mesh.vertices()}
    for vertex in mesh.vertices():
        if distances[vertex] == float("inf"):
            continue
        current = vertex
        while current is not None:
            paths[vertex].insert(0, current)
            current = previous_vertices[current]

    distance_list = list(distances.values())
    print("multi_source_dijkstra", distance_list)
    print("paths", paths)
    return distance_list, paths


# def get_distance_inside_mesh_vertex(mesh:Mesh, source):
#     distance_list,paths=multi_source_dijkstra_with_path(mesh,[source])
#     for i,path in enumerate(paths):
#         pts,distance=line_x_mesh(mesh=mesh,vertex1=path[0],vertex2=path[-1])
#         if distance==0:
#             continue
#         else:
#             distance_list[i]=distance
#     return distance_list

def cube_distances(mesh_input:Mesh,mesh_close:Mesh,source):
    # 将 Compas Mesh 转换为 Trimesh
    time1=time.time()
    vertices, faces = mesh_close.to_vertices_and_faces()
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    tri_mesh.fill_holes()
    #tri_mesh.export(r'C:\Users\Ke Dao\compas_slicer\examples\10_test4_slicing\example_jun_bg\output\tri_closed_mesh.obj')
    bounds = tri_mesh.bounds
    print("cube distance for vertex",source,",Model bounds:", bounds)
    offset_vector=np.array(bounds[0])
    # 定义体素分辨率
    voxel_size = 5

    # 将 Trimesh 网格转换为体素模型
    voxelized = tri_mesh.voxelized(pitch=voxel_size).fill()
 
    # 获取体素模型的实际空间坐标
    voxel_coords = np.array(np.nonzero(voxelized.matrix)).T  # 获取非零体素的坐标
    #print_list_with_details(voxel_coords,"voxel_coords")
    voxel_coords = voxel_coords * voxel_size +offset_vector # 转换为实际空间坐标
    
    #print_list_with_details(voxel_coords,"voxel_coords")
    # 创建一个新的 Trimesh 对象，用体素坐标生成点云
    # voxel_points = trimesh.points.PointCloud(voxel_coords)
    # print(voxel_points,"voxel_points")
    # 将点云保存为 OBJ 文件
    #voxel_points.export(r'C:\Users\Ke Dao\compas_slicer\examples\10_test4_slicing\example_jun_bg\output\voxel_model.obj')

    # 获取体素矩阵
    matrix = voxelized.matrix
    #save_nested_list(nested_list=matrix,file_name="martrix",file_path=r'C:\Users\Ke Dao\compas_slicer\examples\10_test4_slicing\example_jun_bg\output')
    dims = matrix.shape

    # 将目标点坐标转换为体素坐标
    target_point = np.array(mesh_input.vertex_coordinates(source)-offset_vector)  # 替换为你的目标点坐标
    target_voxel = np.round(target_point / voxel_size).astype(int)

    # 定义26个邻居方向
    directions = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
                        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
                        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
                        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]])
    direction_lengths = np.linalg.norm(directions, axis=1)
    # 初始化距离场
    distance_field = np.full(dims, np.inf)
    distance_field[tuple(target_voxel)] = 0

    # 初始化优先队列
    priority_queue = [(0, tuple(target_voxel))]
    # print("start dijstra",matrix,dims,target_voxel)
    # Dijkstra 算法
    while priority_queue:
        current_distance, current_voxel = heapq.heappop(priority_queue)

        if current_distance > distance_field[current_voxel]:
            continue

        for direction, length in zip(directions, direction_lengths):
            neighbor = tuple(np.add(current_voxel, direction))

            if (0 <= neighbor[0] < dims[0] and 
                0 <= neighbor[1] < dims[1] and 
                0 <= neighbor[2] < dims[2] and 
                matrix[neighbor]):
                
                distance = current_distance + length  # 邻居间的距离为方向向量的模长
                #print(current_voxel,distance)

                if distance < distance_field[neighbor]:
                    distance_field[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

    # 打印距离场
    output_field=[]
    for vi in mesh_input.vertices():
        coors=mesh_input.vertex_coordinates(vi)
        coors-=offset_vector
        coors/=voxel_size
        x=int(coors[0])
        y=int(coors[1])
        z=int(coors[2])
        vertex_field=distance_field[x][y][z]
        if np.isinf(vertex_field):
            vertex_field=distance_field[x+1][y][z]
            if np.isinf(vertex_field):
                vertex_field=distance_field[x+1][y+1][z]
                if np.isinf(vertex_field):
                    vertex_field=distance_field[x][y+1][z]
                    if np.isinf(vertex_field):
                        vertex_field=distance_field[x][y+1][z+1]
                        if np.isinf(vertex_field):
                            vertex_field=distance_field[x][y][z+1]
                            if np.isinf(vertex_field):
                                vertex_field=distance_field[x+1][y][z+1]
                                #print(vertex_field)
        output_field.append(vertex_field)
    timeconsume=time.time()-time1
    print_list_with_details(output_field,"output_field")
    print("_time consume",timeconsume)
    return output_field

def cube_distances_multi_sources(mesh_input:Mesh,mesh_close:Mesh,sources):
    # 将 Compas Mesh 转换为 Trimesh
    time1=time.time()
    print("cube distance multi for vetex ",sources[0]," to ",sources[-1]," vertices number ",len(sources))
    vertices, faces = mesh_close.to_vertices_and_faces()
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    tri_mesh.fill_holes()
    #tri_mesh.export(r'C:\Users\Ke Dao\compas_slicer\examples\10_test4_slicing\example_jun_bg\output\tri_closed_mesh.obj')
    bounds = tri_mesh.bounds
    # print("Model bounds:", bounds)
    offset_vector=np.array(bounds[0])
    # 定义体素分辨率
    voxel_size = 5

    # 将 Trimesh 网格转换为体素模型
    voxelized = tri_mesh.voxelized(pitch=voxel_size).fill()
 
    # 获取体素模型的实际空间坐标
    voxel_coords = np.array(np.nonzero(voxelized.matrix)).T  # 获取非零体素的坐标
    #print_list_with_details(voxel_coords,"voxel_coords")
    voxel_coords = voxel_coords * voxel_size +offset_vector # 转换为实际空间坐标
    
    #print_list_with_details(voxel_coords,"voxel_coords")
    # 创建一个新的 Trimesh 对象，用体素坐标生成点云
    # voxel_points = trimesh.points.PointCloud(voxel_coords)
    # print(voxel_points,"voxel_points")
    #save_nested_list(nested_list=voxel_coords,file_name="voxel_points",file_path=r'C:\Users\Ke Dao\compas_slicer\examples\10_test4_slicing\whole\output')
    # 将点云保存为 OBJ 文件
    #voxel_points.export(r'C:\Users\Ke Dao\compas_slicer\examples\10_test4_slicing\example_jun_bg\output\voxel_model.obj')

    # 获取体素矩阵
    matrix = voxelized.matrix
    #save_nested_list(nested_list=matrix,file_name="martrix",file_path=r'C:\Users\Ke Dao\compas_slicer\examples\10_test4_slicing\example_jun_bg\output')
    dims = matrix.shape
    target_voxels=[]
    # 将目标点坐标转换为体素坐标
    # for source in sources:
    #     target_point = np.array(mesh_input.vertex_coordinates(source)-offset_vector)  # 替换为你的目标点坐标
    #     target_voxels.append(np.round(target_point / voxel_size).astype(int))
# 检查源点体素坐标是否在体素矩阵范围内且在体素模型内
    def find_valid_voxel(start_voxel, matrix, dims, found_voxels):
        queue = [start_voxel]
        visited = set()
        
        while queue:
            voxel = queue.pop(0)
            if tuple(voxel) in visited:
                continue
            visited.add(tuple(voxel))
            
            if all(0 <= voxel[i] < dims[i] for i in range(3)) and matrix[tuple(voxel)]:
                if tuple(voxel) not in found_voxels:
                    found_voxels.add(tuple(voxel))
                    return voxel
            
            # 添加邻居方向
            neighbors = [
                voxel + np.array([1, 0, 0]), voxel + np.array([-1, 0, 0]),
                voxel + np.array([0, 1, 0]), voxel + np.array([0, -1, 0]),
                voxel + np.array([0, 0, 1]), voxel + np.array([0, 0, -1])
            ]
            queue.extend(neighbors)
        
        return None  # 如果找不到有效的体素
    print(dims)
    found_voxels = set()
    source_voxels = [np.round(np.array(mesh_input.vertex_coordinates(p)-offset_vector) / voxel_size).astype(int) for p in sources]
    source_voxels = [find_valid_voxel(v, matrix, dims,found_voxels) for v in source_voxels]
    #print_list_with_details(source_voxels,"source_voxels = [v for v in valid_source_voxels if v is not None]")
    source_voxels = [v for v in source_voxels if v is not None]
    #print_list_with_details(source_voxels,"source_voxels = [v for v in source_voxels if v is not None]")

    # 定义26个邻居方向
    directions = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
                        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
                        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
                        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]])
    direction_lengths = np.linalg.norm(directions, axis=1)
    # 初始化距离场
    distance_field = np.full(dims, np.inf)


    # 初始化优先队列
    priority_queue = [(0, tuple(source_voxel)) for source_voxel in source_voxels]

    # print("start dijstra",matrix,dims,source_voxels)
    # Dijkstra 算法
    # 多源 Dijkstra 算法
    while priority_queue:
        current_distance, current_voxel = heapq.heappop(priority_queue)

        if not all(0 <= current_voxel[i] < dims[i] for i in range(3)):
            continue

        if current_distance > distance_field[current_voxel]:
            continue

        for direction, length in zip(directions, direction_lengths):
            neighbor = tuple(np.add(current_voxel, direction))

            if not all(0 <= neighbor[i] < dims[i] for i in range(3)):
                continue

            if matrix[neighbor]:
                distance = current_distance + length  # 使用预计算的方向向量模长

                if distance < distance_field[neighbor]:
                    distance_field[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

    # def get_distance_field_value(x, y, z, distance_field, dims):
    #     if 0 <= x < dims[0] and 0 <= y <= dims[1] and 0 <= z < dims[2]:
    #         return distance_field[x, y, z]
    #     else:
    #         #print(x,y,z,dims)
    #         return np.inf
    output_field=[]
    for vi in mesh_input.vertices():
        coors=mesh_input.vertex_coordinates(vi)
        coors-=offset_vector
        coors/=voxel_size
        x=int(coors[0])
        y=int(coors[1])
        z=int(coors[2])
        vertex_field= get_distance_field_value(x, y, z, distance_field, dims)
        if np.isinf(vertex_field):
            vertex_field = get_distance_field_value(x+1, y, z, distance_field, dims)
            if np.isinf(vertex_field):
                vertex_field = get_distance_field_value(x, y+1, z, distance_field, dims)
                if np.isinf(vertex_field):
                    vertex_field = get_distance_field_value(x, y, z+1, distance_field, dims)
                    if np.isinf(vertex_field):
                        vertex_field = get_distance_field_value(x, y+1, z+1, distance_field, dims)
                        if np.isinf(vertex_field):
                            vertex_field = get_distance_field_value(x+1, y+1, z, distance_field, dims)
                            if np.isinf(vertex_field):
                                vertex_field = get_distance_field_value(x+1, y, z+1, distance_field, dims)
                                if np.isinf(vertex_field):
                                    vertex_field = get_distance_field_value(x+1, y+1, z+1, distance_field, dims)
                                    if np.isinf(vertex_field):
                                        vertex_field = get_distance_field_value(x-1, y, z, distance_field, dims)
                                        if np.isinf(vertex_field):
                                            vertex_field = get_distance_field_value(x, y-1, z, distance_field, dims)
                                            if np.isinf(vertex_field):
                                                vertex_field = get_distance_field_value(x, y, z-1, distance_field, dims)   
                                                if np.isinf(vertex_field):
                                                    vertex_field = get_distance_field_value(x, y-1, z+1, distance_field, dims)
                                                    if np.isinf(vertex_field):
                                                        vertex_field = get_distance_field_value(x-1, y+1, z, distance_field, dims)
                                                        if np.isinf(vertex_field):
                                                            vertex_field = get_distance_field_value(x-1, y, z+1, distance_field, dims)  
                                                            if np.isinf(vertex_field):
                                                                vertex_field = get_distance_field_value(x, y+1, z-1, distance_field, dims)
                                                                if np.isinf(vertex_field):
                                                                    vertex_field = get_distance_field_value(x+1, y-1, z, distance_field, dims)
                                                                    if np.isinf(vertex_field):
                                                                        vertex_field = get_distance_field_value(x+1, y, z-1, distance_field, dims)  
                                                                        if np.isinf(vertex_field):
                                                                            vertex_field = get_distance_field_value(x, y-1, z-1, distance_field, dims)
                                                                            if np.isinf(vertex_field):
                                                                                vertex_field = get_distance_field_value(x-1, y-1, z, distance_field, dims)
                                                                                if np.isinf(vertex_field):
                                                                                    vertex_field = get_distance_field_value(x-1, y, z-1, distance_field, dims)    
                                                                                    if np.isinf(vertex_field):
                                                                                        print(x,y,z,"inf")                                                                    
        output_field.append(vertex_field)
    print_list_with_details(output_field,"output_field")
    print("time consumed",time.time()-time1)
    return output_field

def not_flip_cube_distances_multi_sources(mesh_input:Mesh,sources,derection=True):
    # 将 Compas Mesh 转换为 Trimesh
    vertices, faces =mesh_input.to_vertices_and_faces()
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    bounds = tri_mesh.bounds
    offset_vector=np.array(bounds[0])
    voxel_size = 3
    voxelized = tri_mesh.voxelized(pitch=voxel_size)
    voxel_coords = np.array(np.nonzero(voxelized.matrix)).T  # 获取非零体素的坐标
    voxel_coords = voxel_coords * voxel_size +offset_vector # 转换为实际空间坐标
    matrix = voxelized.matrix

    dims = matrix.shape
    target_voxels=[]

    def find_valid_voxel(start_voxel, matrix, dims, found_voxels):
        queue = [start_voxel]
        visited = set()
        
        while queue:
            voxel = queue.pop(0)
            if tuple(voxel) in visited:
                continue
            visited.add(tuple(voxel))
            
            if all(0 <= voxel[i] < dims[i] for i in range(3)) and matrix[tuple(voxel)]:
                if tuple(voxel) not in found_voxels:
                    found_voxels.add(tuple(voxel))
                    return voxel
            
            # 添加邻居方向
            neighbors = [
                voxel + np.array([1, 0, 0]), voxel + np.array([-1, 0, 0]),
                voxel + np.array([0, 1, 0]), voxel + np.array([0, -1, 0]),
                voxel + np.array([0, 0, 1]), voxel + np.array([0, 0, -1])
            ]
            queue.extend(neighbors)
        
        return None  # 如果找不到有效的体素
    print(dims,"dims not flip")
    found_voxels = set()
    source_voxels = [np.round(np.array(mesh_input.vertex_coordinates(p)-offset_vector) / voxel_size).astype(int) for p in sources]
    source_voxels = [find_valid_voxel(v, matrix, dims,found_voxels) for v in source_voxels]
    print_list_with_details(source_voxels,"source_voxels = [v for v in valid_source_voxels if v is not None]")
    source_voxels = [v for v in source_voxels if v is not None]
    print_list_with_details(source_voxels,"source_voxels = [v for v in source_voxels if v is not None]")

    # 定义26个邻居方向
    if derection:
        directions = np.array([[0, 0, 1], 
                            [1, 0, 1],  [-1, 0, 1], 
                            [0, 1, 1],  [0, -1, 1],
                            [1, 1, 1],  [1, -1, 1],
                            [-1, 1, 1],  [-1, -1, 1],
                            [1, 0, 0],  [-1, 0, 0], 
                            [0, 1, 0],  [0, -1, 0],
                            [1, 1, 0],  [1, -1, 0],
                            [-1, 1, 0],  [-1, -1, 0]                           
                            ])
    else:
        directions = np.array([ [0, 0, -1],
                             [1, 0, -1],  [-1, 0, -1],
                             [0, 1, -1],  [0, -1, -1],
                             [1, 1, -1],  [1, -1, -1],
                             [-1, 1, -1],  [-1, -1, -1],
                             [1, 0, 0],  [-1, 0, 0], 
                            [0, 1, 0],  [0, -1, 0],
                            [1, 1, 0],  [1, -1, 0],
                            [-1, 1, 0],  [-1, -1, 0] 
                             ])        
    #direction_lengths = np.linalg.norm(directions, axis=1)
    # 初始化距离场
    distance_field = np.full(dims, np.inf)


    # 初始化优先队列
    priority_queue = [(0, tuple(source_voxel)) for source_voxel in source_voxels]

    # print("start dijstra",matrix,dims,source_voxels)
    # Dijkstra 算法
    # 多源 Dijkstra 算法
    while priority_queue:
        current_distance, current_voxel = heapq.heappop(priority_queue)

        if not all(0 <= current_voxel[i] < dims[i] for i in range(3)):
            continue

        if current_distance > distance_field[current_voxel]:
            continue

        for direction in directions:
            neighbor = tuple(np.add(current_voxel, direction))

            if not all(0 <= neighbor[i] < dims[i] for i in range(3)):
                continue

            if matrix[neighbor]:
                distance = current_distance   # 使用预计算的方向向量模长

                if distance < distance_field[neighbor]:
                    distance_field[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))


    output_field=[]
    for vi in mesh_input.vertices():
        coors=mesh_input.vertex_coordinates(vi)
        coors-=offset_vector
        coors/=voxel_size
        x=int(coors[0])
        y=int(coors[1])
        z=int(coors[2])
        vertex_field= get_distance_field_value(x, y, z, distance_field, dims)
        if np.isinf(vertex_field):
            vertex_field = get_distance_field_value(x+1, y, z, distance_field, dims)
            if np.isinf(vertex_field):
                vertex_field = get_distance_field_value(x, y+1, z, distance_field, dims)
                if np.isinf(vertex_field):
                    vertex_field = get_distance_field_value(x, y, z+1, distance_field, dims)
                    if np.isinf(vertex_field):
                        vertex_field = get_distance_field_value(x, y+1, z+1, distance_field, dims)
                        if np.isinf(vertex_field):
                            vertex_field = get_distance_field_value(x+1, y+1, z, distance_field, dims)
                            if np.isinf(vertex_field):
                                vertex_field = get_distance_field_value(x+1, y, z+1, distance_field, dims)
                                if np.isinf(vertex_field):
                                    vertex_field = get_distance_field_value(x+1, y+1, z+1, distance_field, dims)
                                    if np.isinf(vertex_field):
                                        vertex_field = get_distance_field_value(x-1, y, z, distance_field, dims)
                                        if np.isinf(vertex_field):
                                            vertex_field = get_distance_field_value(x, y-1, z, distance_field, dims)
                                            if np.isinf(vertex_field):
                                                vertex_field = get_distance_field_value(x, y, z-1, distance_field, dims)   
                                                if np.isinf(vertex_field):
                                                    vertex_field = get_distance_field_value(x, y-1, z+1, distance_field, dims)
                                                    if np.isinf(vertex_field):
                                                        vertex_field = get_distance_field_value(x-1, y+1, z, distance_field, dims)
                                                        if np.isinf(vertex_field):
                                                            vertex_field = get_distance_field_value(x-1, y, z+1, distance_field, dims)  
                                                            if np.isinf(vertex_field):
                                                                vertex_field = get_distance_field_value(x, y+1, z-1, distance_field, dims)
                                                                if np.isinf(vertex_field):
                                                                    vertex_field = get_distance_field_value(x+1, y-1, z, distance_field, dims)
                                                                    if np.isinf(vertex_field):
                                                                        vertex_field = get_distance_field_value(x+1, y, z-1, distance_field, dims)  
                                                                        if np.isinf(vertex_field):
                                                                            vertex_field = get_distance_field_value(x, y-1, z-1, distance_field, dims)
                                                                            if np.isinf(vertex_field):
                                                                                vertex_field = get_distance_field_value(x-1, y-1, z, distance_field, dims)
                                                                                if np.isinf(vertex_field):
                                                                                    vertex_field = get_distance_field_value(x-1, y, z-1, distance_field, dims)    
                                                                                    #if np.isinf(vertex_field):
                                                                                        #print(x,y,z,"inf")                                                                    
        output_field.append(vertex_field)
    print_list_with_details(output_field,"output_field")
    return output_field
def get_distance_field_value(x, y, z, distance_field, dims):
    #if 0 <= x < dims[0] and 0 <= y <= dims[1] and 0 <= z < dims[2]:
    try:
        return distance_field[x, y, z]
    #else:
    except:
        #print(x,y,z,dims)
        return np.inf
def get_close_mesh(mesh:Mesh,cluster_vs_high,cluster_vs_low):
    mesh_close=copy.deepcopy(mesh)

    for cluster_org in cluster_vs_high+cluster_vs_low:
        cluster=rearrange_points(mesh,cluster_org)
        coordinates=[]
        for veky in cluster:
            coordinates.append(np.array(mesh.vertex_coordinates(key=veky)))
        average_coordinates = np.mean(coordinates, axis=0)


        mesh_close.add_vertex(x=average_coordinates[0],y=average_coordinates[1],z=average_coordinates[2])
    
        for i,veky in enumerate(cluster):
            if i < (len(cluster)-1):
                mesh_close.add_face(vertices=[veky,cluster[i+1],mesh_close._max_vertex])
            elif i ==(len(cluster)-1):
                mesh_close.add_face(vertices=[veky,cluster[0],mesh_close._max_vertex])
    return mesh_close

def rearrange_points(mesh:Mesh, points):
    result = []

    # 使用递归函数进行深度优先搜索
    def dfs(current_point, visited:set):
        # 将当前点加入已访问列表
        visited.add(current_point)
        result.append(current_point)
        # 获取当前点的邻居列表
        neighbors = mesh.vertex_neighbors(current_point)

        # 遍历邻居，找到下一个未访问的点
        for neighbor in neighbors:
            if neighbor not in visited and neighbor in points:
                dfs(neighbor, visited)

    # 从第一个点开始进行深度优先搜索
    visited = set()
    dfs(points[0], visited)
    # print(result)
    # print(mesh.edge_faces(u=result[0],v=result[1]))
    if mesh.edge_faces(u=result[0],v=result[1])[1]==None:
        result.reverse()
    return result




def multi_source_dijkstra_cannot_flip(mesh: Mesh, sources, Derection=True):
    print("start_multi_source_dijkstra")
    # 初始化距离字典，所有顶点的距离初始为无穷大
    distances = {vertex: float("inf") for vertex in mesh.vertices()}

    # 优先队列，存储 (距离, 顶点) 对
    priority_queue = []

    # 对于每个源点，将其距离设为0，并加入优先队列
    for source in sources:
        distances[source] = 0
        heapq.heappush(priority_queue, (0, source))

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        # 如果从优先队列中取出的顶点距离大于已知最短距离，则跳过
        if current_distance > distances[current_vertex]:
            continue
        z1 = mesh.vertex_attribute(key=current_vertex, name="z")
        for neighbor in mesh.vertex_neighbors(current_vertex):
            z2 = mesh.vertex_attribute(key=neighbor, name="z")
            if Derection:
                not_flip = z2 >= z1
            else:
                not_flip = z1 >= z2
            #print(not_flip, z1 - z2, Derection)
            if not_flip:
                weigh_vertex = 0.3
                # weigh=mesh.edge_length(current_vertex,neighbor)*(mesh.edge_attribute(edge=(current_vertex,neighbor),name='weigh_cur')*(1-weigh_vertex)+weigh_vertex)
                weigh = mesh.edge_length(current_vertex, neighbor)
            else:
                weigh = float("inf")
            # print(weigh)
            distance = current_distance + weigh  # 假设每条边的权重为1

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    distance_list = list(distances.values())
    print("multi_source_dijkstra", max(distance_list), min(distance_list))
    return distance_list


# def get_distance_inside_mesh_vertex(mesh: Mesh, source):
#     # 初始化距离字典，所有顶点的距离初始为无穷大
#     source_neiborhood=set(mesh.vertex_neighborhood(key=source,ring=2))-set(mesh.vertex_neighborhood(key=source,ring=1))
#     distances = {vertex: float("inf") for vertex in mesh.vertices()}
#     distances[source] = 0

#     # 优先队列，存储 (距离, 顶点) 对
#     priority_queue = [(0, source)]
#     vertex_need_check=set(mesh.vertices())

#     start_time=time.time()
#     print("in mesh timing start",start_time,len(source_neiborhood))
#     time1=start_time

#     while priority_queue:
#         current_distance, current_vertex = heapq.heappop(priority_queue)

#         # 如果从优先队列中取出的顶点距离大于已知最短距离，则跳过
#         if current_distance > distances[current_vertex]:
#             continue
#         vertex_neibors = mesh.vertex_neighbors(current_vertex)
#         vertex_neibors_lenths = [
#                 mesh.edge_length(current_vertex, neighbor)
#                 for neighbor in vertex_neibors
#             ]
        
#         if current_vertex == source or current_vertex in source_neiborhood:
#             print_list_with_details(list(vertex_need_check),"vertex_need_check")
            

#             vertex_ray_neibors, vertex_ray_neibors_lenth = find_ray_neighbors(
#                 mesh, current_vertex,vertex_need_check ,source
#             )
#             vertex_all_neibors = vertex_neibors + vertex_ray_neibors
#             print_list_with_details(vertex_ray_neibors,"vertex_ray_neibors")
#             vertex_all_lenth = vertex_neibors_lenths + vertex_ray_neibors_lenth
#             print_list_with_details(vertex_ray_neibors_lenth,"vertex_ray_neibors_lenth")
#             time2=time.time()
#             print("time comsumed",time2-time1,current_vertex)
#             time1=time2
#         else:
#             vertex_all_neibors = vertex_neibors
#             vertex_all_lenth = vertex_neibors_lenths
        
#         for i, neighbor in enumerate(vertex_all_neibors):
#             distance = current_distance + vertex_all_lenth[i]
#             # print(i,"vertex_all_lenth[i]",vertex_all_lenth[i])

#             if distance < distances[neighbor]:
#                 distances[neighbor] = distance
#                 heapq.heappush(priority_queue, (distance, neighbor))
    
#     distance_list = list(distances.values())
#     end_time=time.time()
#     print_list_with_details(distance_list,"distance_inside_mesh_vertex")
#     print(source,"distance_inside_mesh_vertex"," time using",end_time-start_time)
#     return distance_list

def get_real_distances(mesh:Mesh,source):
    distances=[]
    if isinstance(source,list):
        
        for vertex in mesh.vertices():
            dists=[]
            for so in source:
                distance=compute_distance(mesh=mesh,vertex1=vertex,vertex2=so)
                dists.append(distance)
            distances.append(dists)        
    else:
        for vertex in mesh.vertices():
            distance=compute_distance(mesh=mesh,vertex1=vertex,vertex2=source)
            distances.append(distance)
    return distances


# def find_ray_neibors_org(mesh: Mesh, vertex,vertex_need_check:set, start_vertex=None):
#     ray_neibors = []
#     ray_lenth = []
#     for vertex_i, data in mesh.vertices(data=True):
#         data["ray_neibor"] = None
#     vertex_neibors = mesh.vertex_neighbors(vertex)
#     points_will_check = copy.deepcopy(vertex_need_check)
#     points_will_check -= set(vertex_neibors)
#     points_will_check -= set([vertex])
#     vertex_normal = mesh.vertex_normal(key=vertex)
#     vector_from_start = compute_vector(mesh, start_vertex, vertex)
#     for vertex1 in points_will_check:
#         vector = compute_vector(mesh, vertex, vertex1)
#         if np.dot(vertex_normal, vector) < 0 and (start_vertex==vertex or np.dot(vector_from_start, vector)> 0  ) :
#             data1 = mesh.vertex_attributes(key=vertex1)
#             # print("data1[ray_neibor]",data1["ray_neibor"])
#             if data1["ray_neibor"] == None:
#                 # print(vertex)
#                 intersect_face = ray_x_mesh(
#                     mesh_input=mesh, vertex1=vertex, vertex2=vertex1
#                 )
#                 if intersect_face == []:
#                     data1["ray_neibor"] = True
#                     vertex1_neiborhood=mesh.vertex_neighborhood(key=vertex1,ring=3)
#                     data1s=[mesh.vertex_attributes(key=vertex1_neibor) for vertex1_neibor in vertex1_neiborhood]
#                     for item1n,data1i in enumerate(data1s):
#                         data1i["ray_neibor"] = True
#                         vertex_need_check.discard(vertex1)
#                         ray_neibors.append(vertex1_neiborhood[item1n])
#                         ray_lenth.append(compute_distance(mesh=mesh,vertex1=vertex,vertex2=vertex1_neiborhood[item1n]))

#                     vertex_need_check.discard(vertex1)
#                     ray_neibors.append(vertex1)
#                     ray_lenth.append(compute_distance(mesh=mesh,vertex1=vertex,vertex2=vertex1))
#                 else:
                    
#                     for i, face in enumerate(intersect_face):
#                         corner_vertices = mesh.face_vertices(fkey=face)
#                         if i == 0:
#                             for vertex2 in corner_vertices:
#                                 data2 = mesh.vertex_attributes(key=vertex2)
#                                 if data2["ray_neibor"] == None and vertex2 in vertex_need_check:
#                                     data2["ray_neibor"] = True
#                                     vertex_need_check.discard(vertex2)
#                                     ray_neibors.append(vertex2)
#                                     ray_lenth.append(
#                                         compute_distance(mesh, vertex, vertex2)
#                                     )
#                         else:
#                             for vertex2 in corner_vertices:
#                                 data2 = mesh.vertex_attributes(key=vertex2)
#                                 if data2["ray_neibor"] == None:
#                                     data2["ray_neibor"] = False
#                     if data1["ray_neibor"] == None:
#                         data1["ray_neibor"] = False
        
    #print_list_with_details(ray_lenth,"ray lenth")
    return ray_neibors, ray_lenth

# def find_ray_neighbors(mesh:Mesh, vertex, vertex_need_check: set, start_vertex=None):
#     ray_neighbors = []
#     ray_length = []
    
#     # Initialize ray_neibor attribute for all vertices
#     for vertex_i, data in mesh.vertices(data=True):
#         data["ray_neibor"] = None
    
#     vertex_neighbors = set(mesh.vertex_neighbors(vertex))
#     points_will_check = vertex_need_check - vertex_neighbors - {vertex}
#     vertex_normal = mesh.vertex_normal(key=vertex)
#     vector_from_start = compute_vector(mesh, start_vertex, vertex)
    
#     for vertex1 in points_will_check:
#         vector = compute_vector(mesh, vertex, vertex1)
        
#         if np.dot(vertex_normal, vector) < 0 and (start_vertex == vertex or np.dot(vector_from_start, vector) > 0):
#             data1 = mesh.vertex_attributes(key=vertex1)
            
#             if data1["ray_neibor"] is None:
#                 intersect_face = ray_x_mesh(mesh_input=mesh, vertex1=vertex, vertex2=vertex1)
                
#                 if not intersect_face:
#                     data1["ray_neibor"] = True
#                     vertex1_neighborhood = mesh.vertex_neighborhood(key=vertex1, ring=2)
                    
#                     for neighbor in vertex1_neighborhood:
#                         neighbor_data = mesh.vertex_attributes(key=neighbor)
#                         if neighbor_data["ray_neibor"] is None:
#                             neighbor_data["ray_neibor"] = True
#                             vertex_need_check.discard(neighbor)
#                             ray_neighbors.append(neighbor)
#                             ray_length.append(compute_distance(mesh=mesh, vertex1=vertex, vertex2=neighbor))
                    
#                     vertex_need_check.discard(vertex1)
#                     ray_neighbors.append(vertex1)
#                     ray_length.append(compute_distance(mesh=mesh, vertex1=vertex, vertex2=vertex1))
#                 else:
#                     for i, face in enumerate(intersect_face):
#                         corner_vertices = mesh.face_vertices(fkey=face)
#                         if i == 0:
#                             for vertex2 in corner_vertices:
#                                 data2 = mesh.vertex_attributes(key=vertex2)
#                                 if data2["ray_neibor"] is None and vertex2 in vertex_need_check:
#                                     data2["ray_neibor"] = True
#                                     vertex_need_check.discard(vertex2)
#                                     ray_neighbors.append(vertex2)
#                                     ray_length.append(compute_distance(mesh, vertex, vertex2))
#                         else:
#                             for vertex2 in corner_vertices:
#                                 data2 = mesh.vertex_attributes(key=vertex2)
#                                 if data2["ray_neibor"] is None:
#                                     data2["ray_neibor"] = False
#                     if data1["ray_neibor"] is None:
#                         data1["ray_neibor"] = False
    
#     return ray_neighbors, ray_length

def compute_distance(mesh: Mesh, vertex1, vertex2):
    distance= np.linalg.norm(  
        np.array(mesh.vertex_coordinates(key=vertex1))
        - np.array(mesh.vertex_coordinates(key=vertex2))
    )
    # print(vertex1,vertex2,"distance",distance)
    return distance


def compute_vector(mesh: Mesh, vertex1, vertex2):
    return (np.array(mesh.vertex_coordinates(key=vertex2)) - np.array(
        mesh.vertex_coordinates(key=vertex1))
    )


# def ray_x_mesh_org(mesh_input: Mesh, vertex1, vertex2):

#     #print("tup",vertex1,vertex2,"try")
#     face_distances = intersection_ray_mesh(
#         (
#             np.array(mesh_input.vertex_coordinates(key=vertex1)),
#             (
#                 np.array(mesh_input.vertex_coordinates(key=vertex2))
#                 - np.array(mesh_input.vertex_coordinates(key=vertex1))
#             )
#         ),
#         (mesh_input.to_vertices_and_faces())
#     )
#     faces_org = [t[0] for t in face_distances]
#     faces=[]
#     faces_except=mesh_input.vertex_faces(key=vertex2)
#     for face in faces_org:

#         if face in  faces_except :

#             pass
#         else:

#             faces.append(face)


#     return (faces)

# def ray_x_mesh(mesh_input:Mesh, vertex1, vertex2):
#     # 获取起始点和方向向量
#     start_point = np.array(mesh_input.vertex_coordinates(key=vertex1))
#     direction_vector = np.array(mesh_input.vertex_coordinates(key=vertex2)) - start_point

#     # 获取与网格相交的所有面及其距离
#     face_distances = intersection_ray_mesh(
#         (start_point, direction_vector),
#         mesh_input.to_vertices_and_faces()
#     )
#     faces_org = [t[0] for t in face_distances]

#     # 获取排除的面
#     faces_except = set(mesh_input.vertex_faces(key=vertex2))

#     # 使用集合差集操作进行过滤并保持顺序
#     faces = [face for face in faces_org if face not in faces_except]

#     return faces



# def line_x_mesh(mesh: Mesh, vertex1, vertex2):
#     line = Line(mesh.vertex_coordinates(vertex1), mesh.vertex_coordinates(vertex2))
#     neibor_face = set(mesh.vertex_faces(key=vertex1)) | set(
#         mesh.vertex_faces(key=vertex2)
#     )
#     not_neibor_face = set(mesh.faces()) - neibor_face
#     face_intersetion = {}
#     for face in not_neibor_face:
#         face_intersetion.setdefault(face, {})["intersect"] = False
#         vertices = [l_corner[1] for l_corner in mesh.face_corners(fkey=face)]
#         ver_coor = [mesh.vertex_coordinates(vertex) for vertex in vertices]
#         intersect_point = intersection_line_triangle(line=line, triangle=ver_coor)
#         start = np.array(line.start)
#         if intersect_point:
#             intersect_point = np.array(intersect_point)
#             face_intersetion[face]["intersect"] = True
#             face_intersetion[face]["distance"] = np.linalg.norm(start - intersect_point)
#             face_intersetion[face]["face"] = face
#     filtered_items = {k: v for k, v in face_intersetion.items() if "distance" in v}
#     sorted_items = sorted(filtered_items.items(), key=lambda item: item[1]["distance"])
#     sorted_face_list = [item[1]["face"] for item in sorted_items]
#     print(sorted_face_list)
#     if sorted_face_list == []:
#         return sorted_face_list, line.length
#     else:
#         return sorted_face_list, 0

def print_list_with_details(lst, list_name):
    # 定义一个辅助函数来格式化浮点数
    def format_float(x):
        return f"{x:.2f}" if isinstance(x, float) else x
    
    # 定义一个递归函数来处理嵌套列表
    def format_list(l):
        return [format_list(x) if isinstance(x, list) else format_float(x) for x in l]
    
    # 判断列表长度
    length = len(lst)
    
    # 如果列表长度小于10，则直接打印列表和名称
    if length < 10:
        formatted_list = format_list(lst)
        print(f"{list_name}: {formatted_list}")
    # 如果列表长度大于或等于10，则打印前五个、长度和后五个
    else:
        formatted_list_start = format_list(lst[:5])
        formatted_list_end = format_list(lst[-5:])
        print(f"{list_name} (length: {length}): {formatted_list_start} ... {formatted_list_end}")

def save_nested_list(file_path, file_name, nested_list):
    """
    将嵌套链表保存到JSON文件中。
    
    :param file_path: 保存文件的路径
    :param file_name: 保存文件的名字
    :param nested_list: 要保存的嵌套链表
    """
    # Convert numpy arrays to lists if necessary
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to list
        return obj

    # Convert the nested_list to JSON serializable format
    serializable_list = convert_to_json_serializable(nested_list)

    # Construct the full file path
    full_file_path = os.path.join(file_path, file_name + ".json")

    # Save to JSON file
    with open(full_file_path, 'w') as file:
        json.dump(serializable_list, file)
    print("save_nested_list_",file_name)
def save_nested_list(file_path, file_name, nested_list):
    """
    
    
    :param file_path: 
    :param file_name:
    :param nested_list:
    """
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    full_path = os.path.join(file_path, file_name)
    
    
    with open(full_path, 'w') as file:
        json.dump(nested_list, file)
    print("save_nested_list_",file_name)
def load_nested_list(file_path, file_name):

    full_path = os.path.join(file_path, file_name)
    with open(full_path, 'r') as file:
        nested_list = json.load(file)
    print("load_nested_list_",file_name)
    return nested_list
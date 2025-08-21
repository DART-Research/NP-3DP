from compas_slicer.pre_processing.preprocessing_utils.topological_sorting import SegmentsDirectedGraph
import compas_slicer.utilities as utils
from compas_slicer.geometry import Path
from compas.geometry import distance_point_point, distance_point_point_sqrd
from interpolation_slicer_dart import InterpolationSlicer_Dart
class DirectGraph_dart(SegmentsDirectedGraph):
    def __init__(self, slicer:InterpolationSlicer_Dart, segments, max_d_threshold, DATA_PATH):
        super().__init__(slicer.mesh, segments, max_d_threshold, DATA_PATH)
        self.layer_boundary_inf=slicer.layer_height
        
    
    def find_roots(self):
        """ Roots are vertical_layers_print_data that lie on the build platform. Like that they can be print first. """
        boundary_pts = utils.get_mesh_vertex_coords_with_attribute(self.mesh, 'boundary', 1)
        root_segments = []
        for i, segment in enumerate(self.segments):
            print(i)
            first_curve_pts = segment.paths[0].points
            if are_neighboring_point_clouds(boundary_pts, first_curve_pts, self.max_d_threshold):
                print(self.max_d_threshold)
                root_segments.append(i)
        return root_segments
    
    #################################
# --- helpers

def are_neighboring_point_clouds(pts1, pts2, threshold):
    """
    Returns True if 3 or more points of the point clouds are closer than the threshold. False otherwise.

    Parameters
    ----------
    pts1: list, :class: 'compas.geometry.Point'
    pts2: list, :class: 'compas.geometry.Point'
    threshold: float
    """
    count = 0
    for pt in pts1:
        d = distance_point_point(pt, utils.get_closest_pt(pt, pts2))
        if d < threshold:
            count += 1
            if count > 2:
                return True
    return False


def is_true_mesh_adjacency(all_meshes, key1, key2):
    """
    Returns True if the two meshes share 3 or more vertices. False otherwise.

    Parameters
    ----------
    all_meshes: list, :class: 'compas.datastructures.Mesh'
    key1: int, index of mesh1
    key2: int, index of mesh2
    """
    count = 0
    mesh1 = all_meshes[key1]
    mesh2 = all_meshes[key2]
    pts_mesh2 = [mesh2.vertex_coordinates(vkey) for vkey, data in mesh2.vertices(data=True)
                 if (data['cut'] > 0 or data['boundary'] > 0)]
    for vkey, data in mesh1.vertices(data=True):
        if data['cut'] > 0 or data['boundary'] > 0:
            pt = mesh1.vertex_coordinates(vkey)
            ci = utils.get_closest_pt_index(pt, pts_mesh2)
            if distance_point_point_sqrd(pt, pts_mesh2[ci]) < 0.00001:
                count += 1
                if count == 3:
                    return True
    return False


if __name__ == '__main__':
    pass

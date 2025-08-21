from compas_slicer.print_organization import InterpolationPrintOrganizer
from Topology_Sorting_dart import DirectGraph_dart
from compas_slicer.parameters import get_param
class Print_Organizer_dart(InterpolationPrintOrganizer):
    def __init__(self, slicer, parameters, DATA_PATH):
        super().__init__(slicer, parameters, DATA_PATH)
    def topological_sorting(self):
        """ When the print consists of various paths, this function initializes a class that creates
        a directed graph with all these parts, with the connectivity of each part reflecting which
        other parts it lies on, and which other parts lie on it."""
        #Yichuan's chuange
        #max_layer_height = get_param(self.parameters, key='max_layer_height', defaults_type='layers')
        max_layer_height = get_param(self.parameters, key='avg_layer_height', defaults_type='layers')

        print(self.parameters)

        self.topo_sort_graph = DirectGraph_dart(self.slicer, self.vertical_layers,
                                                               max_layer_height, DATA_PATH=self.DATA_PATH)
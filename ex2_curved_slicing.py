import os
from compas.datastructures import Mesh
import logging
import compas_slicer.utilities as utils
from compas_slicer.slicers import InterpolationSlicer
from gradient_evaluation_dart import GradientEvaluation
from compas_slicer.post_processing import simplify_paths_rdp
from compas_slicer.print_organization import set_extruder_toggle, set_linear_velocity_by_range
from compas_slicer.print_organization import add_safety_printpoints
from compas_slicer.pre_processing import create_mesh_boundary_attributes
from compas_slicer.print_organization import InterpolationPrintOrganizer
from compas_slicer.post_processing import seams_smooth
from compas_slicer.print_organization import smooth_printpoints_up_vectors, smooth_printpoints_layer_heights
import time
from interpolationdart import DartPreprocesssor
from compas.files import OBJWriter
from mesh_cutting import mesh_cutter
from optimizing_offset_value import optimizing_offset_value
from interpolation_slicer_dart import InterpolationSlicer_Dart,simplify_paths_rdp_with_gr,seams_smooth_with_gr
from compas.geometry import Line
from distances import print_list_with_details,cube_distances,get_close_mesh,save_nested_list
from Target_finding import Target_finder
import copy
import pymsgbox
from gradient_optimization import GradientOptimization


logger = logging.getLogger('logger')
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)


input_folder_name='beam1B'
DATA_PATH = os.path.join(os.path.dirname(__file__), input_folder_name)
OUTPUT_PATH = utils.get_output_directory(DATA_PATH)
OBJ_INPUT_NAME = os.path.join(DATA_PATH, 'mesh.obj')

def main():
    start_time = time.time()
    avg_layer_height = 10
    
        

    try:
        
        mesh = Mesh.from_obj(os.path.join(OUTPUT_PATH,"edited_mesh.obj"))  
        print("old mesh")
         
    except:
        print("new mesh")
        mesh = Mesh.from_obj(os.path.join(DATA_PATH, OBJ_INPUT_NAME))
        try:
            print('manual set target')
            low_boundary_vs = utils.load_from_json(DATA_PATH, 'boundaryLOW.json')
            high_boundary_vs = utils.load_from_json(DATA_PATH, 'boundaryHIGH.json')
            create_mesh_boundary_attributes(mesh, low_boundary_vs, high_boundary_vs) 
        except:
            print('auto set target')
            T_f = Target_finder(mesh)
            high_boundary_vs,low_boundary_vs = T_f.output() 
        from mesh_changing import mesh_refiner
        m_r = mesh_refiner(mesh,0.1)
        m_r.change_mesh_shape()
        obj_writer = OBJWriter(filepath= os.path.join(OUTPUT_PATH, "edited_mesh.obj"), meshes=[mesh])
        obj_writer.write()
        mesh= Mesh.from_obj(os.path.join(OUTPUT_PATH,"edited_mesh.obj"))  
        try:
            print('manual set target')
            low_boundary_vs = utils.load_from_json(DATA_PATH, 'boundaryLOW.json')
            high_boundary_vs = utils.load_from_json(DATA_PATH, 'boundaryHIGH.json')
            create_mesh_boundary_attributes(mesh, low_boundary_vs, high_boundary_vs) 
        except:
            print('auto set target')
            T_f = Target_finder(mesh)
            high_boundary_vs,low_boundary_vs = T_f.output() 
        # end
    
    try:
        print('manual set target')
        low_boundary_vs = utils.load_from_json(DATA_PATH, 'boundaryLOW.json')
        high_boundary_vs = utils.load_from_json(DATA_PATH, 'boundaryHIGH.json')
        create_mesh_boundary_attributes(mesh, low_boundary_vs, high_boundary_vs) 
    except:
        print('auto set target')
        T_f = Target_finder(mesh)
        high_boundary_vs,low_boundary_vs = T_f.output() 
    
    parameters = {
        'avg_layer_height': avg_layer_height,  # controls number of curves that will be generated
    }

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
    saddles=preprocessor.find_critical_points(g_eval_height, output_filename= 'height_saddles.json',sort_by_height=True)
    print('number of saddle points:',len(saddles))

      
    # # 求target high 的saddle point
    # g_eval = preprocessor.create_gradient_evaluation(norm_filename='gradient_norm.json', g_filename='gradient.json',
    #                                                  target_2=preprocessor.target_HIGH,way='org')
    
    # saddles_high=preprocessor.find_critical_points(g_eval, output_filename='saddles_high.json')
    # # 求 target low 的 saddle point
    # g_eval = preprocessor.create_gradient_evaluation(norm_filename='gradient_norm.json', g_filename='gradient.json',
    #                                                  target_1=preprocessor.target_LOW,
    #                                                  way='org')
    
    # saddles_low=preprocessor.find_critical_points(g_eval, output_filename='saddles_low.json')
    #print(saddles_low,saddles_high)
    print('saddle points vertex keys:',saddles)
    #slice(OUTPUT_PATH,mesh,preprocessor,saddles,parameters,start_time, sinha=0.12120715136981101**(1+0.5505394107719388)-0.012929264762091174)
    slice(OUTPUT_PATH,mesh,preprocessor,g_eval_height,parameters,start_time,
               sinha=-2,
               animation_frame='',path_on_boundary=True,only_normalize_big_data=False,way='final',g_n=True,cutting=True)#saddle_high=saddles_high,saddle_low=saddles_lowway='final'
    # slice_with_gradient_flow(OUTPUT_PATH,mesh,preprocessor,g_eval_height,parameters,start_time,
    #                         sinha=0.025,
    #                         animation_frame='',path_on_boundary=False,only_normalize_big_data=False,way='z',g_n=True,cutting=False)
    # sinha=0.12120715136981101**(100/100+0.5505394107719388)-0.012929264762091174
    # print(sinha)
    # for animation_frame in range(0,60):
    #     #1/(1.5)**animation_frame
    #     #(0.725)**(animation_frame/91)-0.7
    #     # sinha=0.12120715136981101**(animation_frame/100+0.5505394107719388)-0.012929264762091174
    #     slice(OUTPUT_PATH,mesh,preprocessor,saddles,parameters,start_time,
    #            sinha=sinha,
    #            animation_frame=animation_frame,path_on_boundary=False,only_normalize_big_data=False,way='down',g_n=True)



def slice(OUTPUT_PATH,mesh:Mesh,preprocessor:DartPreprocesssor,g_eval_height:GradientEvaluation,parameters,start_time,sinha=0.025,animation_frame="",g_n=False,path_on_boundary=True,only_normalize_big_data=False,way='final',saddle_high=None,saddle_low=None,cutting=True):
    """
    slicing with the animation frame
    animation_frame: int
    sinha the value for controlling the sharpness of influence field
    file will be save in the original file name+animation_frame
    """
    saddles=g_eval_height.saddles
    print('slice start'+str(animation_frame)+'_________________________')
     
    if way=='up' or way=='down':
        single_side=True
    else:
        single_side=False
    if way != 'z':
        # This part is add by Yichuan cutting the mesh
        offseter=optimizing_offset_value(mesh=mesh,target01=preprocessor.target_LOW,target02=preprocessor.target_HIGH,saddles_ge=g_eval_height,
                        processor=preprocessor,Outputpath=OUTPUT_PATH,sinha=sinha,animation_frame=animation_frame,only_normalize_big_data=only_normalize_big_data,saddle_sides_limit=2,single_side=single_side)
        #cutter.creat_cutting_lines()
        #offseter.analysis_gradient()
        offseter.offset_target_base_on_spatail_interpolation()


    # --- slicing

    # --- gradient optimization
    # GradientOptimizer=GradientOptimization(preprocessor,OUTPUT_PATH)
    # scalar_field=GradientOptimizer.optimize_gradient()

    # G=preprocessor.create_gradient_evaluation(norm_filename='gradient_norm.json', g_filename='gradient.json',
    #                                           scalar_field=scalar_field)

    # --- org
    G=preprocessor.create_gradient_evaluation(norm_filename='gradient_norm.json', g_filename='gradient.json',
                                                     target_1=preprocessor.target_LOW,
                                                     target_2=preprocessor.target_HIGH,way=way,target_index=-1,
                                                     frame=animation_frame,g_n=g_n,show_graph=False,save_output=True)
    
    # preprocessor.load_scalar_field()
    # preprocessor.load_offset_distance()

    preprocessor.save_scalar_field()


    
 
    saddles=preprocessor.find_critical_points(G, output_filename= 'saddles.json')
    preprocessor.save_scalar_field()
    preprocessor.save_offset_distance()
    if cutting:
        cutter=mesh_cutter(mesh=mesh,target01=preprocessor.target_LOW,target02=preprocessor.target_HIGH,saddles=saddles,
                        processor=preprocessor,Outputpath=OUTPUT_PATH,animation_frame=animation_frame,parameters=parameters,G=G)
        cutter.cut_mesh()
        cutter.slice_segment()
    else:



        x=139
        slicer = InterpolationSlicer_Dart(mesh, preprocessor, parameters,path_on_boundary,)#gradient_evaluation=G
        slicer.slice_model(weights_list=[(i+1) / x-0.001 for i in range(x)])  # compute_norm_of_gradient contours
        
        if slicer.gradient_evaluation is not None:
            simplify_paths_rdp_with_gr(slicer, threshold=0.25)
            seams_smooth_with_gr(slicer, smooth_distance=3) 
        # else:
        #     simplify_paths_rdp(slicer, threshold=0.25)
        #     seams_smooth(slicer, smooth_distance=3)        
        slicer.printout_info()
        utils.save_to_json(slicer.to_data(), OUTPUT_PATH, 'curved_slicer'+str(animation_frame)+'.json')

    # # --- Print organizer
    # print_organizer = InterpolationPrintOrganizer(slicer, parameters, DATA_PATH)
    # print_organizer.create_printpoints()

    # smooth_printpoints_up_vectors(print_organizer, strength=0.5, iterations=10)
    # smooth_printpoints_layer_heights(print_organizer, strength=0.5, iterations=5)

    # set_linear_velocity_by_range(print_organizer, param_func=lambda ppt: ppt.layer_height,
    #                              parameter_range=[avg_layer_height*0.5, avg_layer_height*2.0],
    #                              velocity_range=[150, 70], bound_remapping=False)
    # set_extruder_toggle(print_organizer, slicer)
    # add_safety_printpoints(print_organizer, z_hop=10.0)

    # # --- Save printpoints dictionary to json file
    # printpoints_data = print_organizer.output_printpoints_dict()
    # utils.save_to_json(printpoints_data, OUTPUT_PATH, 'out_printpoints.json')

    end_time = time.time()
    print("Total elapsed time", round(end_time - start_time, 2), "seconds,frame:",animation_frame)
    print("______________________________________________________")

def slice_with_gradient_flow(OUTPUT_PATH,mesh:Mesh,preprocessor:DartPreprocesssor,g_eval_height:GradientEvaluation,parameters,start_time,sinha=0.025,animation_frame="",g_n=False,path_on_boundary=True,only_normalize_big_data=False,way='final',saddle_high=None,saddle_low=None,cutting=True):
    single_side=True

    offseter=optimizing_offset_value(mesh=mesh,target01=preprocessor.target_LOW,target02=preprocessor.target_HIGH,saddles_ge=g_eval_height,
                        processor=preprocessor,Outputpath=OUTPUT_PATH,sinha=sinha,animation_frame=animation_frame,only_normalize_big_data=only_normalize_big_data,saddle_sides_limit=2,single_side=single_side)
    offseter.offset_target_base_on_spatail_interpolation()
    G_org=preprocessor.create_gradient_evaluation(norm_filename='gradient_norm.json', g_filename='gradient.json',
                                                     target_1=preprocessor.target_LOW,
                                                     target_2=preprocessor.target_HIGH,way=way,target_index=-1,
                                                     frame=animation_frame,g_n=g_n,show_graph=False,save_output=True)
    G_org.find_critical_points()
    G_org.kill_max_min()
    G_org.find_critical_points()
    G_org.compute_gradient()

    guide_field=G_org.return_scalar_field_dictionary()
    for i in range(2):
        if i==1:
            save=True
        else:
            save=False
        G=preprocessor.create_gradient_evaluation(norm_filename='gradient_norm.json', g_filename='gradient.json',
                                                        target_1=preprocessor.target_LOW,
                                                        target_2=preprocessor.target_HIGH,way='gradient_descent',target_index=-1,
                                                        frame=animation_frame,g_n=g_n,show_graph=False,save_output=save,guide_field=guide_field)
        G_org.find_critical_points()
        G_org.kill_max_min()
        G_org.find_critical_points()
        guide_field=G.return_scalar_field_dictionary()

    preprocessor.save_scalar_field()
    x=70
    slicer = InterpolationSlicer_Dart(mesh, preprocessor, parameters,path_on_boundary,)#gradient_evaluation=G
    slicer.slice_model(weights_list=[(i+1) / x-0.001 for i in range(x)])  # compute_norm_of_gradient contours
    slicer.printout_info()
    utils.save_to_json(slicer.to_data(), OUTPUT_PATH, 'curved_slicer'+str(animation_frame)+'.json')
if __name__ == "__main__":
    main()
    pymsgbox.alert("finish slicing")
    

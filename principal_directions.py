import numpy as np
from scipy.optimize import least_squares
from compas.datastructures import Mesh

def fit_quadratic_surface(points, normals):
    def residuals(params, points, normals):
        A, B, C, D, E, F, G, H, I, J = params
        residuals = []
        for (x, y, z), (nx, ny, nz) in zip(points, normals):
            # Surface equation
            surface_eq = A*x**2 + B*y**2 + C*z**2 + D*x*y + E*x*z + F*y*z + G*x + H*y + I*z + J
            residuals.append(surface_eq)
            # Normal constraints
            grad_x = 2*A*x + D*y + E*z + G
            grad_y = 2*B*y + D*x + F*z + H
            grad_z = 2*C*z + E*x + F*y + I
            residuals.append(grad_x - nx)
            residuals.append(grad_y - ny)
            residuals.append(grad_z - nz)
        return residuals

    # Initial guess for the parameters
    initial_guess = np.zeros(10)
    result = least_squares(residuals, initial_guess, args=(points, normals))
    return result.x



def get_principal_directions_para(params):

    # Parameters of the quadratic surface
    A, B, C, D, E, F, G, H, I, J = params  # Use the fitted parameters here

    # Example point
    #x1, y1, z1 = points[0]  # First point

    # Calculate Hessian matrix at (x1, y1, z1)
    Hessian = np.array([[2*A, D, E],
                        [D, 2*B, F],
                        [E, F, 2*C]])

    # Calculate eigenvalues and eigenvectors of Hessian matrix
    eigenvalues, eigenvectors = np.linalg.eigh(Hessian)

# Find principal curvature directions (corresponding eigenvectors)
    principal_direction1 = eigenvectors[:, 2]  # Corresponds to largest eigenvalue
    principal_direction2 = eigenvectors[:, 0]  # Corresponds to smallest eigenvalue
    return(principal_direction1,principal_direction2)

# Example points and normals
points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [5, 6, 7], [8, 9, 1], [3, 4, 5]])
normals = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 0.1], [0.3, 0.4, 0.5]])


def get_principal_directions(veky_center,mesh:Mesh):
    neibors=mesh.vertex_neighbors(key=veky_center)
    veky_list=[veky_center] + neibors
    coordinates=[mesh.vertex_coordinates(veky) for veky in veky_list]
    normals=[mesh.vertex_normal(key=veky) for veky in veky_list]
    params=fit_quadratic_surface(coordinates,normals)
    return(get_principal_directions_para(params))

# # Fit quadratic surface
# params = fit_quadratic_surface(points, normals)
# print("Fitted parameters:", params)
# principal_direction1,principal_direction2=get_principal_directions()
# # Print principal curvature directions
# print("Principal curvature direction 1 (max curvature direction):", principal_direction1)
# print("Principal curvature direction 2 (min curvature direction):", principal_direction2)

def vector_projection(a, b, c):
    # Compute projections onto b, c, and b x c
    proj_b = np.dot(a, b) / np.dot(b, b) * b
    proj_c = np.dot(a, c) / np.dot(c, c) * c
    
    # Compute projection onto b x c using cross product
    b_cross_c = np.cross(b, c)
    proj_b_cross_c = np.dot(a, b_cross_c) / np.dot(b_cross_c, b_cross_c) * b_cross_c
    
    # Compute the components
    component_b = proj_b
    component_c = proj_c
    component_b_cross_c = proj_b_cross_c
    
    # Compute the orthogonal component (a - sum of projections)
    #orthogonal_component = a - (component_b + component_c + component_b_cross_c)
    
    return np.linalg.norm(component_b), np.linalg.norm(component_c)

def get_verctor_weigh(a,b,c):
    nor_b,nor_c=vector_projection(a,b,c)
    sum_bc=nor_b+nor_c
    nor_b,nor_c=nor_b/sum_bc,nor_c/sum_bc
    return nor_b
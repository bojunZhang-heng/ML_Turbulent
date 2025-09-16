import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import Transolver_Irregular_Mesh
import Transolver_Structured_Mesh_2D
import Transolver_Structured_Mesh_3D

import logging


def get_model(args):
    model_dict = {
        'Transolver_Irregular_Mesh': Transolver_Irregular_Mesh, # for PDEs in 1D space or in unstructured meshes
        'Transolver_Structured_Mesh_2D': Transolver_Structured_Mesh_2D,
        'Transolver_Structured_Mesh_3D': Transolver_Structured_Mesh_3D,
    }
    return model_dict[args.model]
    #return model_dict[args["model"]]

import sys

"""
USAGE
python ./inference/get_criterion_shape.py \
     --eval_target ./data/dataset_shapenet/02958343/731efc7a52841a5a59139efcde1fedcb.txt \
     --eval_source ./data/dataset_shapenet/02958343/c3858a8b73dcb137e3bdba9430565083.txt \
     --logdir Car_unsup \
     --shapenetv1_path /trainman-mount/trainman-storage-e7719e4d-b36c-4bc0-a3b3-e13a2d53f66d/ShapeNetCore.v1 \
"""
sys.path.append("./auxiliary/")
sys.path.append("./extension/")
sys.path.append("./training/")
sys.path.append("./inference/")
sys.path.append("./scripts/")
sys.path.append("./")
import argument_parser
import trainer as t
import useful_losses as loss
import my_utils
from save_mesh_from_points_and_labels import *
import pymesh
import get_shapenet_model
import high_frequencies
import torch
import figure_2_3
import os
import numpy as np
from pytorch_points.utils.pc_utils import save_pts

def get_3D_rot_matrix(axis, angle):
    if axis == 0:
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    if axis == 1:
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [- np.sin(angle), 0, np.cos(angle)]])
    if axis == 2:
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [1, 0, 0]])

def get_model_id(path, isV2):
    if not isV2:
        return os.path.basename(os.path.dirname(path[0]))
    else:
        return os.path.basename(path[:-len("/models/model_normalized.obj")])

def forward(opt):
    """
    Takes an input and a target mesh. Deform input in output and propagate a
    manually defined high frequency from the oinput to the output
    :return:
    """
    my_utils.plant_seeds(randomized_seed=opt.randomize)
    os.makedirs(opt.output_dir, exist_ok=True)

    trainer = t.Trainer(opt)
    trainer.build_dataset_train_for_matching()
    trainer.build_dataset_test_for_matching()
    trainer.build_network()
    trainer.build_losses()
    trainer.network.eval()

    if opt.eval_list and os.path.isfile(opt.eval_list):
        source_target_files = np.loadtxt(opt.eval_list, dtype=str)
        source_target_files = source_target_files.tolist()
        for i, st in enumerate(source_target_files):
            source, target = st
            cat1, fname1 = source.split('/')
            fname1 = os.path.splitext(fname1)[0]
            cat2, fname2 = target.split('/')
            fname2 = os.path.splitext(fname2)[0]
            if len(opt.shapenetv1_path) > 0:
                source_target_files[i] = (os.path.join(opt.shapenetv1_path, cat1, fname1, "model.obj"), os.path.join(opt.shapenetv1_path, cat2, fname2, "model.obj"))
            elif len(opt.shapenetv2_path) > 0:
                source_target_files[i] = (os.path.join(opt.shapenetv2_path, cat1, fname1, "models", "model_normalized.obj"), os.path.join(opt.shapenetv2_path, cat2, fname2, "models", "model_normalized.obj"))
    elif (opt.eval_source != "" and opt.eval_source[-4:] == ".txt") and (opt.eval_target != "" and opt.eval_target[-4:] == ".txt"):
        source_target_files = [(figure_2_3.convert_path(opt.shapenetv1_path, opt.eval_source), figure_2_3.convert_path(opt.shapenetv1_path, opt.eval_target))]

    rot_mat = get_3D_rot_matrix(1, np.pi/2)
    rot_mat_rev = get_3D_rot_matrix(1, -np.pi/2)
    isV2 = len(opt.shapenetv2_path) > 0
    for i, source_target in enumerate(source_target_files):
        basename = get_model_id(source_target[0], isV2) + "-" + get_model_id(source_target[1], isV2)
        path_deformed = os.path.join(opt.output_dir, basename + "-Sab.ply")
        path_source = os.path.join(opt.output_dir, basename + "-Sa.ply")
        path_target = os.path.join(opt.output_dir, basename +"-Sb.ply")

        mesh_path = source_target[0]
        print(mesh_path)
        source_mesh_edge = get_shapenet_model.link(mesh_path)

        mesh_path = source_target[1]
        target_mesh_edge = get_shapenet_model.link(mesh_path)


        print("Deforming source in target")

        source = source_mesh_edge.vertices
        target = target_mesh_edge.vertices

        pymesh.save_mesh_raw(path_source, source, source_mesh_edge.faces, ascii=True)
        pymesh.save_mesh_raw(path_target, target, target_mesh_edge.faces, ascii=True)

        if len(opt.shapenetv2_path) > 0:
            source = source.dot(rot_mat)
            target = target.dot(rot_mat)

        source = torch.from_numpy(source).cuda().float().unsqueeze(0)
        target = torch.from_numpy(target).cuda().float().unsqueeze(0)

        with torch.no_grad():
            source, _, _, _, _ = loss.forward_chamfer(trainer.network, source, target, local_fix=None,
                                                        distChamfer=trainer.distChamfer)

        try:
            source = source.squeeze().cpu().detach().numpy()
            if len(opt.shapenetv2_path) > 0:
                source = source.dot(rot_mat_rev)
            P2_P1_mesh = pymesh.form_mesh(vertices=source, faces=source_mesh_edge.faces)
            pymesh.save_mesh(path_deformed, P2_P1_mesh, ascii=True)

            # print("computing signal tranfer form source to target")
            # high_frequencies.high_frequency_propagation(path_source, path_deformed, path_target)
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()
            path_deformed = path_deformed[:-4] + ".pts"
            save_pts(path_deformed, source.squeeze().cpu().detach().numpy())


if __name__ == '__main__':
    opt = argument_parser.parser()
    forward(opt)

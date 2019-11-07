import pymesh
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(BASE_DIR + "/../")
import normalize_points
import torch

def link(path1):
	"""
	This function takes a path to the orginal shapenet model and subsample it nicely
	"""
	obj1 = pymesh.load_mesh(path1)
	# obj1, info = pymesh.remove_isolated_vertices(mesh)
	obj1, info = pymesh.remove_isolated_vertices(obj1)
	print("Removed {} isolated vertices".format(info["num_vertex_removed"]))
	obj1, info = pymesh.remove_duplicated_vertices(obj1)
	print("Merged {} duplicated vertices".format(info["num_vertex_merged"]))
	obj1, info = pymesh.remove_degenerated_triangles(obj1)
	new_mesh = pymesh.form_mesh(normalize_points.BoundingBox(torch.from_numpy(obj1.vertices)).numpy(), obj1.faces)
	return new_mesh


if __name__ == '__main__':
	main()
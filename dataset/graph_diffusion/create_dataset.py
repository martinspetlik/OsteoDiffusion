import os
#import dgl
import numpy as np
import pyvista as pv
import matplotlib
import torch


def find_faces_with_node(index, cells):
    """Pass the index of the node in question.
    Returns the face indices of the faces with that node."""
    return [i for i, cell in enumerate(cells) if index in cell]

def find_connected_vertices(index, cells):
    """Pass the index of the node in question.
    Returns the vertex indices of the vertices connected with that node."""
    cids = find_faces_with_node(index, cells)
    connected = np.unique(cells[cids].ravel())
    return np.delete(connected, np.argwhere(connected == index))


def create_dataset(bone_type, n_vertices_to_use=None):
    if bone_type == "L4":
        data_dir = "/data/L4"
    elif bone_type == "left_pelvic":
        data_dir = "/data/left_pelvic"

    u_data_path = os.path.join(data_dir, 'u.npy')
    template_data_path = os.path.join(data_dir, 'template.vtk')
    template = pv.read(template_data_path)

    u = np.load(u_data_path)
    print("u[0] ", u[0].shape)

    # Extract vertices and connectivity
    vertices = template.points

    print("vertices ", vertices)

    cells = template.cells.reshape((-1, 5))[:, 1:5]

    print("cells shape ", cells.shape)

    # print("point data ", template.point_data)

    print("vertices ", vertices.shape)

    ver_0 = vertices[0]
    print("ver_0.point_ids ", ver_0)

    print("cell 1 edges", template.get_cell(1).edges[0].GetPoints())

    cell = template.get_cell(1)
    edges = cell.edges
    points = edges[0]
    print("cell ", cell)
    print("edges ", edges)
    print("points ", points)
    print("point ids ", points.point_ids)


    if n_vertices_to_use is not None:
        vertices = vertices[:n_vertices_to_use]

    graph = dgl.DGLGraph()
    graph.add_nodes(len(vertices), {'x': torch.from_numpy(u[0])})
    print("graph.ndata['x'] ", graph.ndata['x'])

    if os.path.exists("/data/{}_edges.npz".format(bone_type)):
        edges = np.load("/data/{}_edges.npz".format(bone_type))["data"]
    else:
        edges = []
        for vertex_id in range(len(vertices)):
            connected = find_connected_vertices(vertex_id, cells)
            for conected_vertes in connected:
                edges.append([vertex_id, conected_vertes])
            print("vertex_id: {}, connected: {} ".format(vertex_id, connected))

        np.savez_compressed(os.path.join("//", "{}_edges".format(bone_type)), data=np.array(edges))

    print("edges ", edges)
    src, dst = zip(*edges)
    # print("src ", src)
    # print("dst ", dst)
    print("len(src) ", len(src))

    graph.add_edges(src, dst)

    print("graph.edges() ", graph.edges())



if __name__ == "__main__":
    create_dataset(bone_type="L4", n_vertices_to_use=1000)

import numpy as np
import vtk
import os

import vtk_utils


def applyCut(plane: vtk.vtkPlane, poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """
    Cut a VTK poly using a plane, and returns the side that aligns with the plane's normal.

    Inputs:
        plane (vtk.vtkPlane): The plane that is used to cut the mesh. The orientation of the plane is important. 
        poly (vtk.vtkPolyData): The mesh to be sliced/cut.
    Outputs:
        cut (vtk.vtkPolyData): The resulting mesh on the side of the plane's normal.
    """
    # Create cut tool
    cut = vtk.vtkClipPolyData()
    # Feed tool
    if vtk.VTK_MAJOR_VERSION <= 5:
        cut.SetInput(poly)
    else:
        cut.SetInputData(poly)
    # Cut
    cut.SetClipFunction(plane)
    cut.Update()
    # Get normal side
    return cut.GetOutput()

def clearOffset(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """
    Sets the origin of the input mesh to be 0,0,z.

    Inputs:
        poly (vtk.vtkPolyData): A mesh.
    Outputs:
        filter (vtk.vtkPolyData): The mesh with 0,0,z origin.
    """
    # Gets the bounds of the mesh
    bounds = poly.GetBounds()
    # Translates the mesh to 0,0.
    transform = vtk.vtkTransform()
    # Note the z coordinate is set to Z. This way there is no translation over z.
    transform.Translate(-bounds[0],-bounds[2],0)
    # Apply the transformation.
    filter = vtk.vtkTransformPolyDataFilter()
    filter.SetTransform(transform)
    filter.SetInputData(poly)
    filter.Update()
    return filter.GetOutput() 

def save(polydata: vtk.vtkPolyData, save_path: str, save_extension: str, generate_uvs: bool, tex: vtk.vtkTexture = None, size:float = None) -> None:
    """
    Saves a mesh given a set of parameters. For instance, "save extension" allows to choose between "STL" or "OBJ" format.
    If the "OBJ" format is selected, but no texture is provided, then the function will default to a debug texture.
    When saving as "OBJ", the user can also choose if the UVs should be generated or not.
    If the UVs are not generated then no texture will be saved.

    Inputs:
        polydata (vtk.vtkPolyData): The mesh to be saved.
        save_path (str): The path where the mesh should be saved.
        tex (vtk.vtkTexture): The texture to attach to this mesh.
        save_extension (str): The extension of the file. Can be either "STL" or "OBJ".
        generate_uvs (bool): Whether or not the UVs should be generated when saving as an "OBJ".
    """
    if save_extension.lower() == "stl":
        vtk_utils.stlWriter(polydata, save_path)
    elif save_extension.lower() == 'obj':
        if tex is None:
            tex = vtk_utils.makeDebugTexture()
        if generate_uvs:
            vtk_utils.objExporter(polydata, save_path, tex, size=size)
        else:
            vtk_utils.objWriter(polydata, save_path, tex)
    else:
        raise ValueError("File extension "+save_extension+" not supported. Choose STL or OBJ instead.")

def cut(polydata: vtk.vtkPolyData, path: str, size: float = 50, tex: np.ndarray = None, save_extension: str = "STL", generate_uvs: bool = False) -> None:
    """
    Cuts a mesh into a smaller meshes of a given size.
    The way this works in by creating a set of planes arranged in an xy grid pattern and cutting along these planes.
    When saving the different parts, the function saves it under "map_xpos_ypos".
    Where "xpos" and "ypos" stand for the x position and y position of that block.
    If a texture is provided, the function will also cut that texture such that it matches the cutted meshes.
    If the "OBJ" format is selected, but no texture is provided, then the function will default to a debug texture.
    When saving as "OBJ", the user can also choose if the UVs should be generated or not.
    If the UVs are not generated then no texture will be saved.

    Inputs:
        polydata (vtk.vtkPolyData): The mesh to be sliced into smaller meshes.
        path (str): The path where the different sliced bits will be saved to.
        size (float): The size of each sliced elements.
        tex (nd.array): The texture attached to that mesh.
        save_extension (str): The extension under which the parts must be saved. Can be either "STL" or "OBJ".
        generate_uvs (bool): Whether or not the UVs should be generated when saving as "OBJ".
    """
    # Creates an empty directory to store the parts of the mesh.
    os.makedirs(path, exist_ok=True)
    # Gets the bounds of the mesh to know infer many planes should be generated.
    bounds = polydata.GetBounds()
    poly_max_x = int(bounds[3])
    poly_max_y = int(bounds[1])
    # Gets the size of the texture to infer how to cut it.
    if not tex is None:
        tex_res_x = tex.shape[0]
        tex_res_y = tex.shape[1]
        tex_res_x_r = 1.0*tex_res_x/poly_max_x
        tex_res_y_r = 1.0*tex_res_y/poly_max_y
    else:
        tex_res_x_r = None
        tex_res_y_r = None

    it = 0
    size = int(size)
    for i in range(size, poly_max_x + size, size):
        # Cut and get left side of the mesh
        plane = vtk_utils.makeVTKPlane((0,i,0),(0,-1,0))
        tmp = applyCut(plane, polydata) # left side of the mesh (a strip).
        # We just got a strip that we are going to cut into cubes.
        for j in range(size, poly_max_y + size, size):
            output = cutAndGetTop(tmp, j)
            tmp = cutAndGetBottom(tmp, j)
            output = cleanBlock(output, size,[i-size, j-size])
            saveBlock(output, tex, [i-size, j-size], path, size, [tex_res_x_r, tex_res_y_r], save_extension, generate_uvs)
            if it%250 == 0:
            	print(it,"parts generated.")
            it += 1
        output = cleanBlock(tmp, size, [i-size, j])
        saveBlock(output, tex, [i-size, j], path, size, [tex_res_x_r, tex_res_y_r], save_extension, generate_uvs)
        if it%250 == 0:
        	print(it,"parts generated.")
        it += 1
        # Cut and get right side of the mesh 
        plane = vtk_utils.makeVTKPlane((0,i,0),(0,1,0))
        polydata = applyCut(plane, polydata)
    tmp = polydata
    # We just got a strip that we are going to cut into cubes.
    for j in range(size, poly_max_y + size, size):
        output = cutAndGetTop(tmp, j)
        tmp = cutAndGetBottom(tmp, j)
        output = cleanBlock(output, size, [i, j-size])
        saveBlock(output, tex, [i, j-size], path, size, [tex_res_x_r, tex_res_y_r], save_extension, generate_uvs)
        if it%250 == 0:
        	print(it,"parts generated.")
        it += 1
    output = cleanBlock(tmp, size, [i, j])
    saveBlock(output, tex, [i, j], path, size, [tex_res_x_r, tex_res_y_r], save_extension, generate_uvs)
    if it%250 == 0:
    	print(it,"parts generated.")
    it += 1

def cutAndGetTop(polydata: vtk.vtkPolyData, position: float) -> vtk.vtkPolyData: 
    plane = vtk_utils.makeVTKPlane((position*1.0,0,0),(-1,0,0))
    return applyCut(plane, polydata) # a cube (ready to be saved).

def cutAndGetBottom(polydata: vtk.vtkPolyData, position: float) -> vtk.vtkPolyData: 
    plane = vtk_utils.makeVTKPlane((position*1.0,0,0),(1,0,0))
    return applyCut(plane, polydata) # a strip (that has to be sliced).

def cleanBlock(polydata, size, position):
    Bounds = polydata.GetBounds()
    # If the deviation is small, then replace the mesh by a flat surface.
    if np.abs(Bounds[-1] - Bounds[-2]) < 0.0015:
        polydata = vtk_utils.makeFlatSurface(position[1],position[0],np.mean([Bounds[-1],Bounds[-2]]),(np.min([Bounds[1]-Bounds[0],size]),np.min([Bounds[3]-Bounds[2],size])))
    # Remove the xy offset, to make the object origin (0,0,z).
    return clearOffset(polydata)

def saveBlock(polydata, tex, position, path, size, tex_res, save_extension, generate_uvs):
    # Cut the textute if one is provided.
    Bounds = polydata.GetBounds()
    if not tex is None:
        chunk_tex = tex[int((position[0])*tex_res[0]):int((position+size)*tex_res[0]),int((position[1])*tex_res[1]):int((position+size)*tex_res[1])]
        chunk_tex = vtk_utils.array2VTKImage(chunk_tex)
        save(polydata, os.path.join(path,'map_'+str(position[0])+'_'+str(position[1])), save_extension, generate_uvs, tex=chunk_tex, size=size)
    else:
        rx = (Bounds[1]-Bounds[0])/size
        ry = (Bounds[3]-Bounds[2])/size
        tex = vtk_utils.makeDebugTexture()
        save(polydata, os.path.join(path,'map_'+str(position[0])+'_'+str(position[1])), save_extension, generate_uvs, tex=tex, size=size)
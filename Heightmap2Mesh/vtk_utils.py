from vtk.util import numpy_support
import numpy as np
import vtk

def makeVTKPlane(p: tuple, normal: tuple) -> vtk.vtkPlane:
    """
    Creates a plane using an origin position (x,y,z) and the normal vector (x,y,z).

    Inputs:
        p (tuple): The origin position of the plane (x,y,z).
        normal (tuple): The vector normal to the plane (x,y,z).
    Outputs:
        plane (vtk.vtkPlane): The resulting plane.
    """
    plane = vtk.vtkPlane()
    plane.SetNormal(normal)
    plane.SetOrigin(p)
    return plane

def mkVtkIdList(it: list) -> vtk.vtkIdList:
    """
    Creates a VTK Id list.

    Inputs:
        it (list): A list of ids (integers).
    Outputs:
        vtk_id_list (vtk.vtkIdList): The resulting vtk list.
    """
    vtk_id_list = vtk.vtkIdList()
    for i in it:
        vtk_id_list.InsertNextId(int(i))
    return vtk_id_list

def makeFlatSurface(px: float, py: float, z: float, size: float) -> vtk.vtkPolyData:
    """
    Creates tiles aligned with z up, given a set of initial positions.
    The generated surface will span the following area:
        (px,py) -> (px+size,py+size)

    Inputs:
        px (float): The x position of the origin of the flat surface.
        py (float): The y position of the origin of the flat surface.
        z (float): The height at which the surface will be generated.
        size (float): The size of the surface.
    Outputs:
        mesh (vtk.vtkPolyData): The generated surface mesh.
    """
    #Array of vectors containing the coordinates of each point
    nodes = np.array([[px, py, z], [px + size[0]//2, py, z], [px+size[0], py, z],
                      [px+size[0], py+size[1]//2, z], [px+size[0], py+size[1], z],
                      [px+size[0]//2, py+size[1], z], [px, py+size[1], z], [px, py+size[1]//2, z],
                      [px+size[0]//2, py+size[1]//2, z]])
    #Array of tuples containing the nodes correspondent of each element
    elements =[(0, 1, 8, 7), (7, 8, 5, 6), (1, 2, 3, 8), (8, 3, 4, 
                        5)]
    #Make the building blocks of polyData attributes
    mesh = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()  
    #Load the point and cell's attributes
    for i in range(len(nodes)):
        points.InsertPoint(i, nodes[i])
    for i in range(len(elements)):
        cells.InsertNextCell(mkVtkIdList(elements[i]))
    #Assign pieces to vtkPolyData
    mesh.SetPoints(points)
    mesh.SetPolys(cells)
    return mesh

def array2VTKImage(image: np.ndarray) -> vtk.vtkImageData:
    """
    Transforms a numpy array into a vtk.ImageData.
    The numpy array can be either float32 or uint8/uint16.

    Inputs:
        image (np.ndarray): a 2D or 3D numpy array. I.e a RGB or Grey scale image.
    Outputs:
        vtk_image (vtk.vtkImageData): a vtkImageData.
    """
    # Build VTK Array
    row_size = image.shape[0]
    col_size = image.shape[1]
    if len(image.shape) > 2:
        #vectorized_image = np.transpose(image, (0,1,2))
        vectorized_image = np.reshape(image, newshape=[-1, image.shape[2]])
    else:
        number_of_pixels = image.size
        vectorized_image = np.reshape(image, (number_of_pixels, 1))
    if (image.dtype == np.float32) or (image.dtype == np.float64):
        vtk_array = numpy_support.numpy_to_vtk(vectorized_image, deep=True,
                            array_type=vtk.VTK_FLOAT)
    else:
        vtk_array = numpy_support.numpy_to_vtk(vectorized_image, deep=True,
                            array_type=vtk.VTK_UNSIGNED_CHAR) 
    # Make a VTK Image
    vtk_image = vtk.vtkImageData()
    if len(image.shape) > 2:
        vtk_image.SetDimensions(col_size, row_size, 1)
    else:
        vtk_image.SetDimensions(col_size, row_size, 1)
    vtk_image.AllocateScalars(vtk_array.GetDataType(), 4)
    vtk_image.GetPointData().GetScalars().DeepCopy(vtk_array)
    return vtk_image

def makeDebugTexture() -> vtk.vtkImageData:
    """
    Creates a texture that can be used to debug the generated meshes.
    Outputs:
        image (vtk.vtkImageData): a VTK image containing the generated image.
    """
    # Generates a debug texture
    tex = np.ones((250, 250, 3), dtype=np.uint8) * 255
    def drawborder(tex, width, color):
        tex[:width,:] = color
        tex[-width:,:] = color
        tex[:,:width] = color
        tex[:,-width:] = color
        return tex
    def drawCross(tex, width, color):
        a = int((tex.shape[0] / 2) - (width / 2))
        b = int((tex.shape[0] / 2) + (width / 2))
        tex[a:b,:] = color
        tex[:,a:b] = color
        return tex
    drawCross(tex, 50, np.array([252,240,250]))
    drawborder(tex, 30, np.array([252,240,250]))
    drawCross(tex, 30, np.array([237,56,189]))
    drawborder(tex, 20, np.array([237,56,189]))
    drawCross(tex, 10, np.array([115,3,191]))
    drawborder(tex, 10, np.array([115,3,191]))
    drawborder(tex, 2, np.array([2,0,31]))
    for i in range(250):
        if i%2==0:
            tex[i] = tex[i] // 2
    image = array2VTKImage(tex)
    return image

def objWriter(polydata: vtk.vtkPolyData, save_path: str, image: vtk.vtkImageData = None) -> None:
    """
    Saves a mesh as an OBJ file. This methods does not generate UVs properly which leads to improper exports.

    Inputs:
        polydata (vtk.vtkPolyData): the mesh to be exported.
        image (vtk.vtkImageData): the image to be saved as the texture of the object.
        save_path (str): the path to the saved file.
    """
    # Write an obj file
    obj_writer = vtk.vtkOBJWriter()
    obj_writer.SetFileName(save_path+'.obj')
    if (image is None):
        obj_writer.SetInputData(polydata)
    else: 
        obj_writer.SetInputData(0, polydata)
        obj_writer.SetInputData(1, image)
    obj_writer.Write()

def objExporter(polydata: vtk.vtkPolyData, save_path: str, image: vtk.vtkImageData = None) -> None:
    """
    Saves a mesh as an OBJ file. Unlike objWriter, this methods generates UVs.

    Inputs:
        polydata (vtk.vtkPolyData): the mesh to be exported.
        image (vtk.vtkImageData): the image to be saved as the texture of the object.
        save_path (str): the path to the saved file.
    """
    # Creates the UV map based on a plane.
    bounds = polydata.GetBounds()
    map_to_plane = vtk.vtkTextureMapToPlane()
    map_to_plane.SetInputData(polydata)
    map_to_plane.SetOrigin(0, 0, 0)
    map_to_plane.SetPoint1(bounds[1], 0, 0)
    map_to_plane.SetPoint2(0, bounds[3], 0)
    # Applies the UV map to a mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(map_to_plane.GetOutputPort())
    # Creates a VTK texture
    texture = vtk.vtkTexture()
    texture.SetInputData(image)
    texture.InterpolateOff()
    texture.RepeatOff()
    # Creates an actor to combine the texture and the mesh.
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetTexture(texture)
    actor.GetProperty().SetAmbientColor([1,1,1])
    actor.GetProperty().SetAmbient(1)
    # Creates a render window to be able to save the OBJ.
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    ren.AddActor(actor)
    ren.SetBackground(1, 1, 1)
    renWin.SetSize(1000, 1000)
    # Saves the OBJ file.
    writer = vtk.vtkOBJExporter()
    writer.SetFilePrefix(save_path)
    writer.SetInput(renWin)
    writer.Write()

def stlWriter(polydata: vtk.vtkPolyData, save_path: str) -> None:
    """
    Saves a mesh as an STL file.

    Inputs:
        polydata (vtk.vtkPolyData): the mesh to be exported.
        save_path (str): the path to the saved file.
    """
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(save_path+'.stl')
    writer.SetInputData(polydata)
    writer.Write()
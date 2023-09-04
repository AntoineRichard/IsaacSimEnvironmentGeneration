import numpy as np
import argparse
import warnings
import vtk
import cv2
import os

import vtk_cutting_tools
import vtk_utils
import config


class HM2Mesh:
    """
    A class that implements everything to transform a dem into a mesh.
    """
    def __init__(self,
                hm_path: str = None,
                save_path: str = None,
                tex_path: str = None,
                z_scale: float = 1.0,
                xy_resolution: float = 1.0,
                save_extension: str = "OBJ",
                generate_uvs: bool = True,
                cut_subtiles: bool = True,
                subtile_size: float = 50,
                **kwargs):
        """
        A class that implements everything to transform a dem into a mesh.

        hm_path (str): The path to the heightmap/dem to be loaded. 
        save_path (str): The path under which the file will be saved.
        tex_path (str): The path to the texture file to be used.
        z_scale (float): The z scaling factor.
        xy_resolution (float): The xy scaling factor.
        save_extension (str): The extension to be used when saving the file.
        generate_uvs (bool): Whether or not the UVs should be generated when saving as OBJ.
        cut_subtiles (bool): Whether or not the mesh should be cutted into smaller meshes.
        subtile_size (float): The size of the smaller meshes.
        """
        self.hm_path = hm_path
        self.tex_path = tex_path
        self.save_path = save_path
        self.save_extension = save_extension
        self.generate_uvs = generate_uvs
        self.xy_resolution = xy_resolution
        self.z_scale = z_scale
        self.cut_subtiles = cut_subtiles
        self.subtile_size = subtile_size
        self.hm_ext = ['png', 'tiff', 'tif', 'PNG', 'TIFF', 'TIF']
        self.tex_ext = ['png', 'tiff', 'tif', 'jpeg', 'jpg', 'PNG', 'TIFF', 'TIF', 'JPEG','JPG']

    def rescale_z(self, hmap:np.ndarray) -> np.ndarray:
        """
        Rescales the map along the z axis.
        Inputs:
            hmap (np.ndarray): the heightmap.
        """
        self.z_offset = hmap.min()*self.z_scale
        return (hmap * self.z_scale) - self.z_offset

    def rescale_xy(self, polydata:vtk.vtkPolyData) -> vtk.vtkPolyData:
        """
        Rescales the mesh along the x and y axes.
        Inputs:
            polydata (vtk.vtkPolyData): The mesh to be rescaled.
        Outputs:
            r_polydata (vtk.vtkPolyData): The rescaled mesh.
        """
        # Rescale along the xy dimmensions
        scale = (1/self.xy_resolution, 1/self.xy_resolution, 1)
        transform = vtk.vtkTransform()
        transform.Scale(scale)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputData(polydata)
        transformFilter.SetTransform(transform)
        transformFilter.Update()
        r_polydata = transformFilter.GetOutput()
        return r_polydata
    
    def undo_z_offset(self, polydata:vtk.vtkPolyData) -> vtk.vtkPolyData:
        """
        Undo the offset applied in rescale_z function
        Inputs:
            polydata (vtk.vtkPolyData): The mesh to be rescaled.
        Outputs:
            r_polydata (vtk.vtkPolyData): The un z offseted mesh.
        """
        offset = (0,0,self.z_offset)
        transform = vtk.vtkTransform()
        transform.Translate(offset)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputData(polydata)
        transformFilter.SetTransform(transform)
        transformFilter.Update()
        r_polydata = transformFilter.GetOutput()
        return r_polydata



    def readHeightMap(self) -> np.ndarray:
        """
        Loads the heightmap and returns it as a numpy array.

        Outputs:
            heightmap (np.ndarray): a 2D numpy array containing the elevation map.
        """
        extension = self.hm_path.split('.')[-1]
        # Check the map exists.
        assert os.path.exists(self.hm_path) == True, "The path, "+self.hm_path+", is invalid. Please check the map exists."
        # Load the map based on the extension
        if extension == "npy":
            heightmap = np.load(self.hm_path)
        elif extension in self.hm_ext:
            heightmap = cv2.imread(self.hm_path, cv2.IMREAD_UNCHANGED)
        else:
            raise ValueError("Heigtmap format: "+extension+" not supported.")
        # Check the dimmension is OK.
        # Ideally it should be 2, but 3 is OK.
        if (len(heightmap.shape) > 2):
            if (len(heightmap.shape) > 3):
                raise ValueError("The heightmap dimmension is too large: "+str(len(heightmap.shape))+", expected 2 or 3.")
            if (heightmap.shape[-1] != 1):
                warnings.warn("Number of channel was: "+str(heightmap.shape[-1])+", used the first one.")
            heightmap = heightmap[:,:,0]
        # Check the data type.
        # Ideally it should be float32, but float64 or uint16 is OK.
        if (heightmap.dtype != np.float32) and (heightmap.dtype != np.float64):
            warnings.warn("Loading non floating point map.")
        if (heightmap.dtype == np.uint8):
             warnings.warn("Loading uint8 map is not recommended. The quality of the generated mesh is going to be low.")
        return np.swapaxes(heightmap,0,1)

    def readTexture(self) -> np.ndarray:
        """
        Loads a texture and returns it as a numpy array.

        Output:
            texture (np.array): a 2D numpy array containing the elevation map.
        """
        extension = self.hm_path.split('.')[-1]
        # Check the map exists.
        assert os.path.exists(self.tex_path) == True, "The path, "+self.tex_path+", is invalid. Please check the map exists."
        # Load the map based on the extension
        if extension == "npy":
            texture = np.load(self.tex_path)
        elif extension in self.tex_ext:
            texture = cv2.imread(self.tex_path, cv2.IMREAD_UNCHANGED)
        else:
            raise ValueError("Texture format: "+extension+" not supported.")
        # Check the dimmension is OK.
        # It should be 3.
        if (len(texture.shape) > 3):
            raise ValueError("The texture dimmension is too large: "+str(len(texture.shape))+", expected 3.")
        # No data-type check.
        return np.swapaxes(texture,0,1)
    
    def generateHeightmap(self, image: vtk.vtkImageData) -> vtk.vtkPolyData:
        """
        Creates a mesh from a heightmap using the wrap function. This is the simplest
         way of creating a vtk.vtkPolyData from a heightmap.
        
        Input:
            image (vtk.vtkImageData): A VTK image containing the heightmap.
        Output:
            (vtk.vtkPolyData): A VTK polydata representing the heightmap.
        """
        # Create surface from image
        surface = vtk.vtkImageDataGeometryFilter()
        surface.SetInputData(image)
        # Warp the surface in the vertical direction
        warp = vtk.vtkWarpScalar()
        warp.SetInputConnection(surface.GetOutputPort())
        warp.SetScaleFactor(1)
        warp.UseNormalOn()
        warp.SetNormal(0,0,1)
        warp.Update()
        return warp.GetOutput()

    def run(self):
        """
        Applies all the methods to generate the mesh from the dem and potential texture.
        """
        hmap = self.readHeightMap()
        hmap = self.rescale_z(hmap)
        hmap = vtk_utils.array2VTKImage(hmap)
        tex = None
        if not self.tex_path is None:
            tex = self.readTexture()
            tex = vtk_utils.array2VTKImage(tex)
        else:
            tex = vtk_utils.makeDebugTexture()
        polydata = self.generateHeightmap(hmap)
        polydata = self.rescale_xy(polydata)
        polydata = self.undo_z_offset(polydata)
        if self.save_extension.lower() == "stl":
            vtk_utils.stlWriter(polydata, save_path=self.save_path)
        elif self.save_extension.lower() == "obj":
            if self.generate_uvs:
                vtk_utils.objExporter(polydata, self.save_path, image=tex)
            else: 
                vtk_utils.objWriter(polydata, self.save_path, image=tex)
        else:
            raise ValueError("File extension "+self.save_extension+" not supported. Choose STL or OBJ instead.")
        if self.cut_subtiles:
            if self.tex_path is None:
                vtk_cutting_tools.cut(polydata, self.save_path, save_extension=self.save_extension, generate_uvs=self.generate_uvs, size=self.subtile_size)
            else:
                tex = self.readTexture()
                vtk_cutting_tools.cut(polydata, self.save_path, tex=tex, save_extension=self.save_extension, generate_uvs=self.generate_uvs, size=self.subtile_size)


class HM2MeshCell(HM2Mesh):
    """
    A cell based decimation method to transform a dem into a mesh.
    Inherits from HM2Mesh.
    """
    def __init__(self,
                 hm_path: str = None,
                 save_path: str = None,
                 tex_path: str = None,
                 num_cells: tuple = (0, 0),
                 fxfy: tuple = (-1, -1),
                 z_scale: float = 1.0,
                 xy_resolution: float = 1.0,
                 save_extension: str = "OBJ",
                 generate_uvs: bool = True,
                 cut_subtiles: bool = True,
                 subtile_size: float = 50,
                 **kwargs):
        super().__init__(hm_path,
                         save_path,
                         tex_path,
                         z_scale,
                         xy_resolution,
                         save_extension,
                         generate_uvs,
                         cut_subtiles,
                         subtile_size,
                         **kwargs)
        """
        A class that implements everything to transform a dem into a mesh.
        It uses cell decimation to reduced the size of the generated mesh.

        hm_path (str): The path to the heightmap/dem to be loaded. 
        save_path (str): The path under which the file will be saved.
        tex_path (str): The path to the texture file to be used.
        z_scale (float): The z scaling factor.
        xy_resolution (float): The xy scaling factor.
        save_extension (str): The extension to be used when saving the file.
        generate_uvs (bool): Whether or not the UVs should be generated when saving as OBJ.
        num_cells (tuple): The number of cells to be used on the x and y axis when using cell or point decimation.
        fxfy (tuple): The the ratio of celles to be used on the x and y axis when using cell or point decimation. 
        cut_subtiles (bool): Whether or not the mesh should be cutted into smaller meshes.
        subtile_size (float): The size of the smaller meshes.
        """
        self.num_cells = num_cells
        self.fxfy = fxfy

    def generateHeightmap(self, image: vtk.vtkImageData) -> vtk.vtkPolyData:
        """
        Creates a mesh from a heightmap using the wrap function. This is method uses
        cell approximation to create vtk.vtkPolyData from a heightmap.
        The way it works is by creating a grid of a given size (parametrized by the user).
        Then, the height of the DEM is being probed to assign the correct height to each
        element of the grid.
        
        Input:
            image (vtk.vtkImageData): A VTK image containing the heightmap.
        Output:
            (vtk.vtkPolyData): A VTK polydata representing the heightmap.
        """
        # Generate plane
        zLevel = image.GetBounds()[5]
        demBounds = image.GetBounds()
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(demBounds[0], demBounds[2], zLevel)
        plane.SetPoint1(demBounds[1], demBounds[2], zLevel)
        plane.SetPoint2(demBounds[0], demBounds[3], zLevel)
        # Get the dimension reduction
        if self.num_cells == (0,0):
            assert self.fxfy[0] != 0, "fxfy[0] value cannot be zero."
            assert self.fxfy[1] != 0, "fxfy[1] value cannot be zero."
            if self.fxfy[0] == -1:
                res0 = int(demBounds[1])                
            else:
                res0 = int(demBounds[1]*self.fxfy[0])
            if self.fxfy[1] == -1:
                res1 = int(demBounds[3])                
            else:
                res1 = int(demBounds[3]*self.fxfy[1])
        else:
            assert self.num_cells[0] != 0, "num_cell[0] value cannot be zero."    
            assert self.num_cells[1] != 0, "num_cell[1] value cannot be zero."    
            if self.num_cells[0] == -1:
                res0 = int(demBounds[1])                
            else:
                res0 = self.num_cells[0]
            if self.num_cells[1] == -1:
                res1 = int(demBounds[3])
            else:
                res1 = self.num_cells[1]
        plane.SetResolution(res0, res1)
        plane.Update()
        # Get the scalars from the DEM
        probeDem = vtk.vtkProbeFilter()
        probeDem.SetSourceData(image)
        probeDem.SetInputConnection(plane.GetOutputPort())
        probeDem.Update()
        # Fit polygons to surface (cell strategy)
        cellFit = vtk.vtkFitToHeightMapFilter()
        cellFit.SetInputConnection(probeDem.GetOutputPort())
        cellFit.SetHeightMapData(image)
        cellFit.SetFittingStrategyToCellAverageHeight()
        cellFit.UseHeightMapOffsetOn()
        cellFit.Update()
        return cellFit.GetOutput()


class HM2MeshPoints(HM2Mesh):
    """
    A point based decimation method to transform a dem into a mesh.
    Inherits from HM2Mesh.
    """
    def __init__(self,
                 hm_path: str = None,
                 save_path: str = None,
                 tex_path: str = None,
                 num_cells: tuple = (0, 0),
                 fxfy: tuple = (-1, -1),
                 z_scale: float = 1.0,
                 xy_resolution: float = 1.0,
                 save_extension: str = "OBJ",
                 generate_uvs: bool = True,
                 cut_subtiles: bool = True,
                 subtile_size: float = 50,
                 **kwargs):
        super().__init__(hm_path,
                         save_path,
                         tex_path,
                         z_scale,
                         xy_resolution,
                         save_extension,
                         generate_uvs,
                         cut_subtiles,
                         subtile_size,
                         **kwargs)
        """
        A class that implements everything to transform a dem into a mesh.
        It uses point decimation to reduced the size of the generated mesh.

        hm_path (str): The path to the heightmap/dem to be loaded. 
        save_path (str): The path under which the file will be saved.
        tex_path (str): The path to the texture file to be used.
        z_scale (float): The z scaling factor.
        xy_resolution (float): The xy scaling factor.
        save_extension (str): The extension to be used when saving the file.
        generate_uvs (bool): Whether or not the UVs should be generated when saving as OBJ.
        num_cells (tuple): The number of cells to be used on the x and y axis when using cell or point decimation.
        fxfy (tuple): The the ratio of celles to be used on the x and y axis when using cell or point decimation. 
        cut_subtiles (bool): Whether or not the mesh should be cutted into smaller meshes.
        subtile_size (float): The size of the smaller meshes.
        """
        self.num_cells = num_cells
        self.fxfy = fxfy

    def generateHeightmap(self, image: vtk.vtkImageData) -> vtk.vtkPolyData:
        """
        Creates a mesh from a heightmap using the wrap function. This is method uses
        point approximation to create vtk.vtkPolyData from a heightmap.
        The way it works is by creating a grid of a given size (parametrized by the user).
        Then, the height of the DEM is being probed to assign the correct height to each
        element of the grid.
        
        Input:
            image (vtk.vtkImageData): A VTK image containing the heightmap.
        Output:
            (vtk.vtkPolyData): A VTK polydata representing the heightmap.
        """
        # Generate plane
        zLevel = image.GetBounds()[5]
        demBounds = image.GetBounds()
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(demBounds[0], demBounds[2], zLevel)
        plane.SetPoint1(demBounds[1], demBounds[2], zLevel)
        plane.SetPoint2(demBounds[0], demBounds[3], zLevel)
        # Get the dimension reduction
        if self.num_cells == (0,0):
            assert self.fxfy[0] != 0, "fxfy[0] value cannot be zero."
            assert self.fxfy[1] != 0, "fxfy[1] value cannot be zero."
            if self.fxfy[0] == -1:
                res0 = int(demBounds[1])                
            else:
                res0 = int(demBounds[1]*self.fxfy[0])
            if self.fxfy[1] == -1:
                res1 = int(demBounds[3])                
            else:
                res1 = int(demBounds[3]*self.fxfy[1])
        else:
            assert self.num_cells[0] != 0, "num_cell[0] value cannot be zero."    
            assert self.num_cells[1] != 0, "num_cell[1] value cannot be zero."    
            if self.num_cells[0] == -1:
                res0 = int(demBounds[1])                
            else:
                res0 = self.num_cells[0]
            if self.num_cells[1] == -1:
                res1 = int(demBounds[3])
            else:
                res1 = self.num_cells[1]
        plane.SetResolution(res0, res1)
        plane.Update()
        # Get the scalars from the DEM
        probeDem = vtk.vtkProbeFilter()
        probeDem.SetSourceData(image)
        probeDem.SetInputConnection(plane.GetOutputPort())
        probeDem.Update()
        # Fit polygons to surface (point strategy)
        pointFit = vtk.vtkFitToHeightMapFilter()
        pointFit.SetInputConnection(probeDem.GetOutputPort())
        pointFit.SetHeightMapData(image)
        pointFit.SetFittingStrategyToPointProjection()
        pointFit.UseHeightMapOffsetOn()
        pointFit.Update()
        return pointFit.GetOutput()


class HM2MeshGreedy(HM2Mesh):
    """
    A greedy decimation method to transform a dem into a mesh.
    Inherits from HM2Mesh.
    """
    def __init__(self,
                 hm_path: str = None,
                 save_path: str = None,
                 tex_path: str = None,
                 z_scale: float = 1.0,
                 xy_resolution: float = 1.0,
                 save_extension: str = "OBJ",
                 generate_uvs: bool = True,
                 absolute_error: float = 0.1,
                 cut_subtiles: bool = True,
                 subtile_size: float = 50,
                 **kwargs):
        super().__init__(hm_path,
                         save_path,
                         tex_path,
                         z_scale,
                         xy_resolution,
                         save_extension,
                         generate_uvs,
                         cut_subtiles,
                         subtile_size,
                         **kwargs)
        """
        A class that implements everything to transform a dem into a mesh.
        It uses greedy decimation to reduced the size of the generated mesh.

        hm_path (str): The path to the heightmap/dem to be loaded. 
        save_path (str): The path under which the file will be saved.
        tex_path (str): The path to the texture file to be used.
        z_scale (float): The z scaling factor.
        xy_resolution (float): The xy scaling factor.
        save_extension (str): The extension to be used when saving the file.
        generate_uvs (bool): Whether or not the UVs should be generated when saving as OBJ.
        absolute_error (float): The maximum absolute error when using greedy decimation.
        cut_subtiles (bool): Whether or not the mesh should be cutted into smaller meshes.
        subtile_size (float): The size of the smaller meshes.
        """
        self.absolute_error = absolute_error

    def generateHeightmap(self, image: vtk.vtkImageData) -> vtk.vtkPolyData:
        """
        Creates a mesh from a heightmap using the wrap function. This is method uses
        greedy terrain decimation to create vtk.vtkPolyData from a heightmap.
        The way it works is decimating the geometry as much as possible while matching a maximum
        error constraint.

        Input:
            image (vtk.vtkImageData): A VTK image containing the heightmap.
        Output:
            (vtk.vtkPolyData): A VTK polydata representing the heightmap.
        """
        # Decimate the heightmap
        deci = vtk.vtkGreedyTerrainDecimation()
        deci.SetInputData(image)
        deci.BoundaryVertexDeletionOn()
        deci.SetErrorMeasureToAbsoluteError()
        deci.SetAbsoluteError(self.absolute_error)
        deci.Update()
        return deci.GetOutput()


class HM2MeshFactory:
    """
    The factory for the heighmap/dem to mesh converters.
    """
    def __init__(self):
        self.creators = {}
    
    def register(self, name: str, class_: HM2Mesh) -> None:
        """
        Adds a new converter to the list of available converters.
        Inputs:
            name (str): The name to give to the converter.
            class (HM2Mesh): The class to add (The converter).
        """
        self.creators[name] = class_
        
    def get(self, mode: str="default", **kwargs:dict) -> HM2Mesh:
        """
        Returns an initialized converter.
        Inputs:
            mode (str): The name of the converter to be returned.
            kwargs (dict): The arguments to be used to instantiate the converter.
        """
        return self.creators[mode](**kwargs)


factory = HM2MeshFactory()
factory.register("default", HM2Mesh)
factory.register("cell", HM2MeshCell)
factory.register("point", HM2MeshPoints)
factory.register("greedy", HM2MeshGreedy)


if __name__ == "__main__":
    # Command line example:
    #  
    # python3 heightmap_to_meshes.py\
    # --dems /home/antoine/Downloads/dem_rect.npy\
    # --save_paths /home/antoine/Downloads/test_code\
    # --textures /home/antoine/Downloads/tex_rect.npy\
    # --xy_resolution 10\
    # --decimation_mode greedy\
    # --save_extension OBJ\
    # --generate_uvs 1\
    # --cut_subtiles 1\
    # --subtile_size 10\
    # --absolute_error 0.05

    # Load arguments.
    args, unknown_args = config.parseArgs()
    # Check arguments.
    config.checkArguments(args)
    # Get the configuration.
    cfg = config.loadFromArgs(args)
    # Get the convertion object.
    H2T = factory.get(args.decimation_mode, **cfg.__dict__)

    # Run the desired configuration.
    if args.textures is None:
        it = zip(args.dems, args.save_paths, [None]*len(args.dems))
    elif len(args.textures) != len(args.dems):
        it = zip(args.dems, args.save_paths, args.textures*len(args.dems))
    else:
        it = zip(args.dems, args.save_paths, args.textures)

    for dem, save_path, texture in it:
        # It would be nicer to implement getters and setters for these parameters.
        H2T.hm_path = dem
        H2T.save_path = save_path
        H2T.tex_path = texture
        H2T.run()
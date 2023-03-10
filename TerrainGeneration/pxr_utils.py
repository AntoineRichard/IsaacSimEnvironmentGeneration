import omni
import os
import numpy as np
from pxr import UsdGeom, Gf, Sdf, UsdPhysics, UsdShade, Usd, Vt
from omni.physx.scripts import utils

def loadStage(path: str):
    omni.usd.get_context().open_stage(path)

def saveStage(path: str):
    omni.usd.get_context().save_as_stage(path, None)

def newStage():
    omni.usd.get_context().new_stage()

def closeStage():
    omni.usd.get_context().close_stage()

def setDefaultPrim(stage, path):
    prim = stage.GetPrimAtPath(path)
    stage.SetDefaultPrim(prim)

def movePrim(path_from, path_to):
    omni.kit.commands.execute('MovePrim',path_from=path_from, path_to=path_to)

def createXform(stage, path):
    prim_path = omni.usd.get_stage_next_free_path(stage, path, False)
    obj_prim = stage.DefinePrim(prim_path, "Xform")
    return obj_prim, prim_path

def loadTexture(stage, mdl_path, mdl_name, scene_path):
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name=mdl_path,
        mtl_name=mdl_name,
        mtl_created_list=[os.path.join(scene_path,mdl_name)],
    )
    mtl_prim = stage.GetPrimAtPath(os.path.join(scene_path,mdl_name))
    material = UsdShade.Material(mtl_prim)
    return material

def applyMaterial(prim, material):
    UsdShade.MaterialBindingAPI(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)

def createObject(prefix,
    stage,
    path,
    position=Gf.Vec3d(0, 0, 0),
    rotation=Gf.Rotation(Gf.Vec3d(0,0,1), 0),
    scale=Gf.Vec3d(1,1,1),
    is_instance=True,
) -> tuple:
    """
    Creates a 3D object from a USD file and adds it to the stage.
    """
    obj_prim, prim_path = createXform(stage, prefix)
    obj_prim.GetReferences().AddReference(path)
    if is_instance:
        obj_prim.SetInstanceable(True)
    xform = UsdGeom.Xformable(obj_prim)
    setScale(xform, scale)
    setTransform(xform, getTransform(rotation, position))
    return obj_prim, prim_path

def addCollision(stage, path, mode="none"):
    # Checks that the mode selected by the user is correct.
    accepted_modes = ["none", "convexHull", "convexDecomposition", "meshSimplification", "boundingSphere", "boundingCube"]
    assert mode in accepted_modes, "Decimation mode: "+mode+" for colliders unknown."
    # Get the prim and add collisions.
    prim = stage.GetPrimAtPath(path)
    utils.setCollider(prim, approximationShape=mode)

def deletePrim(stage, path):
    # Deletes a prim from the stage.
    stage.RemovePrim(path)

def createStandaloneInstance(stage, path):
    # Creates and instancer.
    instancer = UsdGeom.PointInstancer.Define(stage, path)
    return instancer

def createInstancerAndCache(stage, path, asset_list):
    # Creates a point instancer
    instancer = createStandaloneInstance(stage, path)
    # Creates a Xform to cache the assets to.
    # This cache must be located under the instancer to hide the cached assets.
    createXform(stage, os.path.join(path,'cache'))
    # Add each asset to the scene in the cache.
    for asset in asset_list:
        # Create asset.
        prim, prim_path = createObject(os.path.join(path,'cache','instance'), stage, asset)
        # Add this asset to the list of instantiable objects.
        instancer.GetPrototypesRel().AddTarget(prim_path)
    # Set some dummy parameters
    setInstancerParameters(stage, path, pos=np.zeros((1,3))) 

def setInstancerParameters(stage, path, pos, ids = None, scale = None, quat = None):
    num = pos.shape[0]
    instancer_prim = stage.GetPrimAtPath(path)
    num_prototypes = len(instancer_prim.GetRelationship("prototypes").GetTargets())
    # Set positions.
    instancer_prim.GetAttribute("positions").Set(pos)
    # Set scale.
    if scale is None:
        scale = np.ones_like(pos)
    instancer_prim.GetAttribute("scales").Set(scale)
    # Set orientation.
    if quat is None:
        quat = np.zeros((pos.shape[0],4))
        quat[:,0] = 1
    instancer_prim.GetAttribute("orientations").Set(quat)
    # Set ids.
    if ids is None:
        ids=  (np.random.rand(num) * num_prototypes).astype(int)
    # Compute extent.
    instancer_prim.GetAttribute("protoIndices").Set(ids)
    updateExtent(stage, instancer_prim)
    
def updateExtent(stage, instancer_path):
    # Get the point instancer.
    instancer = UsdGeom.PointInstancer.Get(stage, instancer_path)
    # Compute the extent of the objetcs.
    extent = instancer.ComputeExtentAtTime(Usd.TimeCode(0), Usd.TimeCode(0))
    # Applies the extent to the instancer.
    instancer.CreateExtentAttr(Vt.Vec3fArray([
        Gf.Vec3f(extent[0]),
        Gf.Vec3f(extent[1]),
    ]))

def enableSmoothShade(prim, extra_smooth=False):
    # Sets the subdivision scheme to smooth.
    prim.GetAttribute("subdivisionScheme").Set(UsdGeom.Tokens.catmullClark)
    # Sets the triangle subdivision rule.
    if extra_smooth:
        prim.GetAttribute("triangleSubdivisionRule").Set(UsdGeom.Tokens.smooth)
    else:
        prim.GetAttribute("triangleSubdivisionRule").Set(UsdGeom.Tokens.catmullClark)

def getTransform(
    rotation: Gf.Rotation,
    position: Gf.Vec3d,
) -> Gf.Matrix4d:
    matrix_4d = Gf.Matrix4d().SetTranslate(position)
    matrix_4d.SetRotateOnly(rotation)
    return matrix_4d

def setProperty(
    xform: UsdGeom.Xformable,
    value,
    property,
) -> None:
    op = None
    for xformOp in xform.GetOrderedXformOps():
        if xformOp.GetOpType() == property:
            op = xformOp
    if op:
        xform_op = op
    else:
        xform_op = xform.AddXformOp(
            property,
            UsdGeom.XformOp.PrecisionDouble,
            "",
        )
    xform_op.Set(value)

def setScale(
    xform: UsdGeom.Xformable,
    value,
) -> None:
    setProperty(xform, value, UsdGeom.XformOp.TypeScale)

def setTranslate(
    xform: UsdGeom.Xformable,
    value,
) -> None:
    setProperty(xform, value, UsdGeom.XformOp.TypeTranslate)

def setRotateXYZ(
    xform: UsdGeom.Xformable,
    value,
) -> None:
    setProperty(xform, value, UsdGeom.XformOp.TypeRotateXYZ)

def setTransform(
    xform: UsdGeom.Xformable,
    value: Gf.Matrix4d,
) -> None:
    setProperty(xform, value, UsdGeom.XformOp.TypeTransform)
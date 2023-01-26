import numpy as np
import argparse
import shutil
import os

def genTag(tag_name, content, content_type="tag", extra=None): 
    xml = {} 
    xml["tag_name"] = tag_name 
    xml["tag_type"] = content_type 
    xml["content"] = content 
    if not extra is None: 
        xml["extra"] = (extra[0], extra[1])
    return xml 

def genXML(data, depth=0): 
    tabs = "".join(["    " for i in range(depth)]) 
    if type(data) is list: 
        xml = [] 
        for dat in data: 
            xml += genXML(dat, depth=depth) 
    elif type(data) is dict: 
        if "extra" in data.keys(): 
            start = tabs+"<"+data["tag_name"]+" "+data["extra"][0]+"=\""+data["extra"][1]+"\">" 
        else: 
            start = tabs+"<"+data["tag_name"]+">" 
        if data["tag_type"] == "tag": 
            content = genXML(data["content"], depth=depth+1) 
            end = tabs+"</"+data["tag_name"]+">" 
            xml = [start] + content + [end] 
        else: 
            content = genXML(data["content"]) 
            end = "</"+data["tag_name"]+">" 
            xml = [start + content + end] 
    elif type(data) is str: 
        xml = data 
    return xml 

def genWorld(names):
    includes = []
    for name in names:
        uri = genTag("uri","model://"+name,"line")
        name = name.split("_")
        pose = genTag("pose",str(name[2])+" "+str(name[1])+" 0 0 0 0", "line")
        includes.append(genTag("include",[uri, pose]))
    uri = genTag("uri","model://sun","line")
    includes.append(genTag("include",uri))
    inc = genTag("world", includes, extra=("name","vineyard"))
    world = genTag("sdf", inc, extra=("version","1.4"))
    header = ["<?xml version=\"1.0\" ?>"]
    return header + genXML(world)

def writeFile(file_name, data):
    with open(file_name, "w") as f:
        for line in data:
            f.write(line+"\n")

def generateSDF(name, local_collision_path, local_visualization_path):
    a = genTag("mu", "2.316343837943928", "line")
    b = genTag("mu2", "2.316343837943928", "line")
    c = genTag("ode", [a,b])
    d = genTag("friction", "2.316343837943928", "line")
    e = genTag("friction2", "2.316343837943928", "line")
    f = genTag("bullet", [d,e])
    g = genTag("friction", [c,f])
    h = genTag("surface", g)
    i = genTag("uri", local_collision_path, "line")
    j = genTag("mesh", i)
    k = genTag("geometry", j)
    l = genTag("collision", [h,k], extra=("name",name+"_collision"))
    m = genTag("uri", local_visualization_path, "line")
    n = genTag("mesh", m)
    o = genTag("geometry", n)
    p = genTag("visual", o, extra=("name",name+"_visual"))
    q = genTag("link", [p,l], extra=("name",name+"_link"))
    r = genTag("static", "True", "line")
    s = genTag("model", [q,r], extra=("name",name))
    t = genTag("sdf", s, extra=("version","1.4"))
    header = ["<?xml version=\"1.0\" ?>"]
    xml = header + genXML(t)
    return xml

def generateConfig(name):
    a = genTag("name", "Antoine Richard", "line")
    b = genTag("producer", "VTK 9.2", "line")
    c = genTag("author", [a,b])
    d = genTag("description", "A portion of a heightmap.", "line")
    e = genTag("sdf ", "model.sdf", "line", extra=("version","1.4"))
    f = genTag("name", name, "line")
    g = genTag("model", [f,e,c,d])
    header = ["<?xml version=\"1.0\" ?>"]
    xml = header + genXML(g)
    return xml

def makeModel(input_path, output_path, name):
    model_path = os.path.join(output_path, name)
    mesh_path = os.path.join(model_path, "meshes")
    viz_mesh_path = os.path.join(mesh_path, "visual")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(mesh_path, exist_ok=True)
    os.makedirs(viz_mesh_path, exist_ok=True)
    os.makedirs(os.path.join(model_path, "thumbnails"), exist_ok=True)
    shutil.copy(os.path.join(input_path, name + ".obj"), os.path.join(viz_mesh_path, name + ".obj"))
    shutil.copy(os.path.join(input_path, name + ".mtl"), os.path.join(viz_mesh_path, name + ".mtl"))
    shutil.copy(os.path.join(input_path, name + "texture1.png"), os.path.join(viz_mesh_path, name + "texture1.png"))
    model = generateSDF(name, os.path.join("meshes","visual", name + ".obj"), os.path.join("meshes", "visual", name + ".obj"))
    config = generateConfig(name)
    writeFile(os.path.join(output_path, name, "model.sdf"), model)
    writeFile(os.path.join(output_path, name, "model.config"), config)
    
def convert_assets(folders, load_materials):
    ext_list = ["obj", "stl", "dae"]
    for in_folder in folders:
        out_folder = in_folder + "_SDF"
        os.makedirs(out_folder, exist_ok=True)
        files = os.listdir(in_folder)
        names = []
        for f in files:
            name = ".".join(f.split(".")[:-1])
            ext = f.split(".")[-1] 
            if ext.lower() in ext_list:
                makeModel(in_folder, out_folder, name)
                names.append(name)
        world = genWorld(names)
        writeFile(os.path.join(out_folder,out_folder+".world"), world)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert OBJ/STL assets to USD")
    parser.add_argument("--folders", type=str, nargs="+", default=None, help="List of folders to convert (space seperated).")
    parser.add_argument("--load_materials", action="store_true", help="If specified, materials will be loaded from meshes")
    args, unknown_args = parser.parse_known_args()

    if args.folders is None:
        raise ValueError(f"No folders specified via --folders argument")

    convert_assets(args.folders, args.load_materials)

import bpy
import math
import csv

# Clear existing scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Load seismic CSV data (time, lat, lon, depth, magnitude)
csv_file = "earthquake_data.csv"

# Convert lat/lon to XYZ using a simple Earth sphere projection
def latlon_to_xyz(lat, lon, radius=10):
    x = radius * math.cos(math.radians(lat)) * math.cos(math.radians(lon))
    y = radius * math.cos(math.radians(lat)) * math.sin(math.radians(lon))
    z = radius * math.sin(math.radians(lat))
    return x, y, z

# Create Earth sphere
bpy.ops.mesh.primitive_uv_sphere_add(radius=10, location=(0, 0, 0))
earth = bpy.context.object
earth.name = "Earth"
bpy.ops.object.shade_smooth()

# Add basic material for Earth
mat_earth = bpy.data.materials.new(name="EarthMaterial")
mat_earth.use_nodes = True
bsdf = mat_earth.node_tree.nodes["Principled BSDF"]
bsdf.inputs['Base Color'].default_value = (0.1, 0.2, 0.6, 1)  # Blue-ish
earth.data.materials.append(mat_earth)

# Load earthquake events
with open(csv_file, newline='') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        depth = float(row["depth"])
        mag = float(row["mag"])
        time = i * 3  # spacing in frames

        x, y, z = latlon_to_xyz(lat, lon, radius=10 - (depth * 0.01))

        bpy.ops.mesh.primitive_uv_sphere_add(radius=mag * 0.1, location=(x, y, z))
        quake = bpy.context.object
        quake.name = f"Quake_{i}"

        mat = bpy.data.materials.new(name=f"QuakeMat_{i}")
        mat.use_nodes = True
        quake.data.materials.append(mat)
        quake.keyframe_insert(data_path="scale", frame=time)
        quake.scale = (0, 0, 0)
        quake.keyframe_insert(data_path="scale", frame=time + 10)

# Set camera
bpy.ops.object.camera_add(location=(0, -25, 15), rotation=(math.radians(60), 0, 0))
bpy.context.scene.camera = bpy.context.object

# Add lighting
bpy.ops.object.light_add(type='SUN', location=(10, 10, 20))

# Set animation settings
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = time + 20

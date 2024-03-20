import sys
import numpy as np
from PIL import Image
from pyvox.parser import VoxParser

def voxel_to_stacked_sprite(voxel_file, output_file):
    # Parse the .vox file
    vox_parser = VoxParser(voxel_file)
    vox_model = vox_parser.parse()

    # Convert the vox model to dense format
    dense_voxels = vox_model.to_dense()

    # Extract the size and voxel data
    size_x, size_y, size_z = dense_voxels.shape

    # Generate the stacked sprite
    sprite = np.zeros((size_z * size_y, size_x, 4), dtype=np.uint8)

    for z in range(size_z):
        for x in range(size_x):
            for y in range(size_y):
                color_idx = dense_voxels[x, y, z]
                if color_idx != 0:
                    sprite[y * size_z + z, x, :] = np.array(vox_model.palette[color_idx], dtype=np.uint8)

    # Save the stacked sprite as a PNG image with a transparent background
    img = Image.fromarray(sprite, "RGBA")
    img.save(output_file)
    return size_y

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python voxel_to_stacked_sprite.py <input_voxel_file> <output_png_file>")
    else:
        voxel_file = sys.argv[1]
        output_file = sys.argv[2]
        layers = voxel_to_stacked_sprite(voxel_file, output_file)
        print( 'Conversion complete! Sprite has %d layers.' % layers )



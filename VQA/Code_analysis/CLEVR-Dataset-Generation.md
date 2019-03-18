## CLEVR Dataset Generation

### Indexing
- [Implementation Code](#Implementation-Code)
- [Rendering Overview]
- [References](#References)
---
### Implementation Code
#### Generating Images
- Code
```
blender --background --python render_images.py -- --num_images 10
```

#### Generating Questions
- Code
```
python generate_questions.py
```
---
### Rendering Overview
data/base_scene.blend
- Contains a **Blender scene** used for the basis of all CLEVR images.
- This scene contains a **ground plane**, a **camera**, and several **light sources**.

Camera and lights
- After loading the base scene, the positions of the camera and lights are randomly jittered
- Controlled with the **--key_light_jitter**, **--fill_light_jitter**, **--back_light_jitter**, **--camera_jitter**

Objects
- After loading the base scene, objects are pkaced one by one into the scene.
- Number of objects for each scene is a random integer between **--min_objects(default 3)**, **--max_objects(default 10)**
- After placing all objects, we ensure that no objects are fully occluded; 
- minimal occupy for each object is 100 pixels (customizable using **--min_pixels_per_object**).
- **To accomplish this, we assign each object a unique color and render a version of the scene with lighting and shading disabled, writing it to a temporary file; we can then count the number of pixels of each color in this pre-render to check the number of visible pixels for each object.**

Other parameters
- **--num_images**: image number to be rendered
- **--start_idx**: start number

#### Object Placement
- Object is positioned randomly
- The minimal distance between object is defined by **--min_dist**
- For each pair of objects, the left/right and front/back distance along the ground plane is at least **--margin** units; this helps to minimize ambiguous spatial relationships.
- If after **--max_retries** attempts we are unable to find a suitable position for an object, then all objects are deleted and placed again from scratch.


#### Image Resolution
- default images are rendered at 320 \* 240
- can be customized using **--height** and **--width**

#### GPU Acceleration
- **--use_gpu 1** to enabling the use GPU
- Currently not support multi-GPUs

#### Rendering Quality
- can be controlled by **--render_num_samples**

#### Saving Blender Scene Files
- can save a Blender .blend file for each rendered image by adding the flag **--save_blendfiles 1**
- each file will be around 5MB

#### Output Files
- Rendered images are stored in the **--output_image_dir** directory
- JSON file for each scene containing ground truth object positions and attributs is saved in the **--output_scene_dir**


#### Object Properties
- **--properties_json** file defines the allowed shapes, sizes, colors and materials used for objects, making it easy to extend CLEVR with new object properties.
- definition of colors
- definition of sizes
- definition of materials

#### Restricting Shape / Color Combinations
- **--shape_color_combos_json** flag can be used to restrict the colors of each shape.


---
### References
- CLEVR Dataset Generation Code(https://github.com/facebookresearch/clevr-dataset-gen#clevr-dataset-generation)
- CLEVR Paper(https://arxiv.org/pdf/1612.06890.pdf)




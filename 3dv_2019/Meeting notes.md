# Day 1: Tutorial - Active 3D Imaging Systems: An In-Depth Look

## Basic Measurement Principles
### Time of flight
- scalable
- range up to kilometers

### Tiangulation
- Spot projection
- Line projection
- Area projection

**Baseline**

## Active Triangulation
- Controlled illumination or observation from multiple viewpoints

### Laser Triangulation
- simplicity, robustness, maturity
- costy

### General Principle
- Triangulation between a camera and a projector
- Binary coding
- Discrete coding: gray code
- A lot of interesting codes exists: binary reflected gray code, larg-gap gray code
- Continuous coding: 
- Phase-shift
- Spatial coding
- Active stereo and phase-shift

## Measurement Uncertainty and Error
- A lof of sources of uncertainty
- Inaccurate comes from projector rather than the camera

- Lateral Resolution
- Speckle and Point-based System
- Dichromatic Model

- Interreflection

## Error mitigation 1 
### Discontinuity Artefacts
- Dual Sensor Compensation
- Space-Time analysis
- Quasi-Analog Projection

### Motion Artifacts
- Heavily dependent on technique

## Error mitigation 2: multiple reflections
- Coding schemes robust to indirect/global illumination
- Code ensemble method
- Micro phase shifting method
- Scene adaptive approaches
- Light Paths Identification (specialized hardware)

## Integrating 3D Imageing Errors
- Pose estimation using UNet
- Sensor-specific artifacts need to be integrated
- Generating training data -> using sensor error charactors for data augmentation

## Advanced Design
- Autosynchronized scanning
- Dual aperture mask + triangulation
- Concealed patterns structured light

## Conclusion
- Acquisition; Processing; Application
- trends (commodification especially tof; miniaturisation; direct gains from ots component improvements)
- Take home message: known your sensors


# Day 2: Keynote speak
## Speaker
- Andrew Davison
- Contents: from SLAM to Spatial AI


- SLAM enabled products and systems
- Cumulative levels of SLAM: robust localization; dense mapping; semantic understanding

### Examples of Spatial AI system
- Robot
- AR glasses

### A large gap to close
- Despite rapid progress, there is still a long way to go

### History - PhD work
- AIST?
- MonoSLAM: sparse feature-based slam (Used a strange, old-style visualization system; augmented for a simple AR)
- Dyson 360 Eye: visual monoslam using omni-directional camera; dyson have big ambitions in home robotics; impressive 
- Dense SLAM: PTAM, DTAM, Kinect, Kinect-fusion
- Elastic Fusion: 
- Semantic Fusion: combines elastic fusion with semantic segmentation network (ICRA 2017)
- Semantic SLAM is important for spatial AI

- SLAM meets Deep Learning: a sliding scale between end-to-end model and human-designed algorithm

### New representations for spatial AI
- CodeSLAM (CVPR 2018)
- SceneCode (CVPR 2019)

- SLAM + Object Recognition (3D Scene representation at object level)
- Fusion++ (3DV2018): Used mask R-CNN
- Event Cameras: toward SLAM competences

- SLAMBench project (Graphcore-IPU(not CPU or GPU; graph; suited for SLAM))


### Spatial AI brain
- camra processors
- camera interfaces
- sensor interface
- actuator interface
- real-time loop
- map store

# Day3: Progress in Unsupervised Domain Alignment with GANs

## GAN and recent develpments
- Latent variables 
- BigGAN
- StyleGAN; AdaIN (feature wise linear transformation)
- SPADE: (normalization, alpha, beta)


## CycleGAN
- Transform across domain without pair-wised data
- General GAN loss + cycle consistence loss (L1 reconstruction)
- Cycle gan seems to be doing pixel-level retexturing
- Stochastic CycleGAN: add noise (determistic to stochastic -- shoes and edges)
- Augmented CycleGAN: augment cycleGAN with latent variables

## ICCV2019: batch weight for domain adaptation with mass shift
- S1 and S2 forms clear clusters, but have different frequencies
- using batch weight (///)
- compared JDBW with MUNIT (munit does style transfer, not really domain adaptation)
- Unsupervised semantic coupling: minist to svhn

## 3D Structure from single ianmge data
- Generates 3D models from 2D images
- HoloGAN (Uses initial 3D CNN blocks; transforms features in the 3D feature spaces; projects to 2D convolutional layers)











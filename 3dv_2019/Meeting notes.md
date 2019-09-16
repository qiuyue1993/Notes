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

















# positioning-circle-from-stereocam

This program is intended to positioning a circle in 5DOF (except the rotation about z-axis) with a stereo camera.

The positioning is separated into two steps:
- find a rough circle with 5 DOF, by using a method that described in the paper:

    [Long Quan. Conic Reconstruction and Correspondence from Two Views. IEEE Transactions on Pattern Analysis and Machine Intelligence, Institute of Electrical and Electronics Engineers, 1996, 18 (2), pp.151--160. ⟨10.1109/34.481540⟩.](https://ieeexplore.ieee.org/abstract/document/481540/)
- refine the circle by edges, which is described in the paper:

    [Malassiotis, S., & Strintzis, M. G. (2003). Stereo vision system for precision dimensional inspection of 3D holes. Machine Vision and Applications, 15(2), 101-113.](https://link.springer.com/article/10.1007/s00138-003-0132-3)

# TODO
- Try to do pre-processing on the edge image: 
    - propose a measure method to see how many edge points are on the fitted ellipse
    - try to connect the separated ellipse's edge component, by maximizing the edge points on the fitted ellipse. 
- Measure the error of the rough positioning method
# positionning-circle-from-sterocam

This program is intended to positionning a circle in 5DOF (except the rotation about z-axis) with a stereo camera.

The positionning is seperated into two steps:
- find roughly circle with 5 DOF, by using a method that described in the paper:

    [Long Quan. Conic Reconstruction and Correspondence from Two Views. IEEE Transactions on Pattern Analysis and Machine Intelligence, Institute of Electrical and Electronics Engineers, 1996, 18 (2), pp.151--160. ⟨10.1109/34.481540⟩.](https://ieeexplore.ieee.org/abstract/document/481540/)
- refine the circle by edges, which is described in the paper:

    [Malassiotis, S., & Strintzis, M. G. (2003). Stereo vision system for precision dimensional inspection of 3D holes. Machine Vision and Applications, 15(2), 101-113.](https://link.springer.com/article/10.1007/s00138-003-0132-3)
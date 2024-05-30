GrabCut is a computer vision algorithm used for image segmentation, particularly for separating the foreground from the background in an image. 
It is an interactive tool based on iterative graph cuts, where the user provides an initial bounding box around the object of interest. 
The algorithm then refines this selection by modeling the foreground and background as Gaussian Mixture Models (GMMs) and iteratively optimizing the segmentation using min-cut/max-flow algorithms on a graph representation of the image.
GrabCut is widely used in applications requiring precise object extraction from images.

I worked on this assignment from scratch based on varius articles i found online as part of the course Fundamentals of Graphics, Image Processing, and Visualization during my second year of studying in Tel-Aviv University. 
This project combines knowledge from Algorithms and Computer Vision and is written in python.

results on chosen examples:
![image](https://github.com/eladsalama/grabcut/assets/100277534/997227f4-a6a5-4623-b278-08962dbe1665)
![image](https://github.com/eladsalama/grabcut/assets/100277534/ebcdc1bb-4963-46a5-9395-641386c01f70)

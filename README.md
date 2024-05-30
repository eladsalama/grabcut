GrabCut is a computer vision algorithm used for image segmentation, particularly for separating the foreground from the background in an image. 
It is an interactive tool based on iterative graph cuts, where the user provides an initial bounding box around the object of interest. 
The algorithm then refines this selection by modeling the foreground and background as Gaussian Mixture Models (GMMs) and iteratively optimizing the segmentation using min-cut/max-flow algorithms on a graph representation of the image.
GrabCut is widely used in applications requiring precise object extraction from images.
![image](https://github.com/eladsalama/grabcut/assets/100277534/997227f4-a6a5-4623-b278-08962dbe1665)
![image](https://github.com/eladsalama/grabcut/assets/100277534/c73513c1-33a3-40ec-a0fa-5ab9742b64fb)


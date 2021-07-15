## Assignment 1 @ COL783 IIT Delhi

# Color-Transfer
Implementation of https://www.researchgate.net/publication/220183710_Transferring_Color_to_Greyscale_Imagesfor transferring color to greyscale images from color images.

* Global_matching.py 
  * Implements the Global Image Matching algorithm as given in the paper.
  ![Color_Image](https://github.com/AyushAniket/Color-Transfer/blob/master/color_global_2.jpg?raw=true) + 
  ![Grey_Image](https://github.com/AyushAniket/Color-Transfer/blob/master/gray_global_2.jpg?raw=true) =
  ![Result_Image](https://github.com/AyushAniket/Color-Transfer/blob/master/result_global_2.png?raw=true)
 
* Swatches.py 
  * Transfers the color by first selecting swatches(by user - as two opposite diagonal points) on color and corresponding greyscale images.
  * After performing global image matching on the selected swatch area , color is distributed by texture synthesis throught the greyscale image.
  ![Color_Image](https://github.com/AyushAniket/Color-Transfer/blob/master/source_swatch.png?raw=true) + 
  ![Grey_Image](https://github.com/AyushAniket/Color-Transfer/blob/master/target_swatch.png?raw=true) =
  ![Result_Image](https://github.com/AyushAniket/Color-Transfer/blob/master/result_swatches.png?raw=true)

# Color-Quantization
Implementation of https://www.cse.iitd.ac.in/~pkalra/col783/assignment1/p297-heckbert.pdf for quantization of colored images.

* Popularity_algorithm.py
  * Selects top K colors out of all the distinct color.
  * Create a colormap of K popular colors and using K_D tree Algorithm for color mapping.

* Median_cut.py
  * The color space is split into K cubes and mean of each cube is the Kth representative in the colormap.

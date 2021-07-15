## Assignment 1 @ COL783 IIT Delhi

# Color-Transfer
Implementation of https://www.researchgate.net/publication/220183710_Transferring_Color_to_Greyscale_Imagesfor transferring color to greyscale images from color images.

* Global_matching.py 
  * Implements the Global Image Matching algorithm as given in the paper.
  ![alt text](https://github.com/AyushAniket/Color-Transfer/blob/master/color.jpg?raw=true)
 
* Swatches.py 
  * Transfers the color by first selecting swatches(by user - as two opposite diagonal points) on color and corresponding greyscale images.
  * After performing golbal image matching on the selected swatch area , color is distributed by texture synthesis throught the greyscale image.
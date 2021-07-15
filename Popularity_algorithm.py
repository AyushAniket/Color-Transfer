
from scipy.spatial import KDTree
import cv2
import numpy as np
import math

def main():
    
    '''Image color quantization using Popularity algorithm for colormap and dithering
    '''
    
    #reading image 
    image = cv2.imread('sample_image.png')      

    #displaying image
    cv2.imshow('image',image)
    print("Press any key to continue")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #converting to RGB color space and data type float32 for numerical operations
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('float32')

    #flattening the image into 1-D array
    y,x = image[:,:,0].shape
    flattened_img_array = image.reshape([y*x,3])

    #getting index and count of unique colors in the image
    values,index, counts = np.unique(flattened_img_array, return_index=True, return_counts=True)

 
    #number of K representatives
    num_K = 4
    
    #sorting the index array with count of unique colors
    sorted_index = index[counts.argsort()]

    l_i = sorted_index.shape[0]

    #index of K popular colors
    colormap_index = sorted_index[l_i-num_K:]

    colormap = np.zeros((num_K,3))

    #creating colormap with index values
    for i in range(len(colormap_index)):
        
        ii = colormap_index[i]

        y_index = int(math.floor(ii/x))
        x_index = ii - x*y_index
        
        colormap[i] = image[y_index,x_index,:] 


    #Using K_D tree Algorithm for color mapping
    tree = KDTree(colormap)

    quantized_image = np.zeros((image.shape))
    y,x = image[:,:,0].shape
    
    #color mapping
    for i in range(x):
        for j in range(y):

            pixel = image[j,i,:]
            
            _,ii = tree.query(image[j,i,:],k=1)
            quantized_image[j,i,:] = colormap[ii]


    #converting the datatype to uint8 for image display
    quantized_image = quantized_image.astype("uint8")

    cv2.imwrite('result_popularity.png',cv2.cvtColor(quantized_image, cv2.COLOR_RGB2BGR))

    #displaying quantized image
    cv2.imshow('quantized_image',cv2.cvtColor(quantized_image, cv2.COLOR_RGB2BGR))

    print("Press any key to exit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == '__main__':
    
    main()





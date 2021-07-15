
from scipy.spatial import KDTree
import cv2
import numpy as np
import math

def main():
    
    '''Image color quantization using Median cut algorithm for colormap and dithering
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

    #list for colormaps
    colormap = []

    #flattening the image into 1-D array 
    y,x = image[:,:,0].shape
    flattened_img_array = image.reshape([y*x,3])

    
    def K_representative(img_arr):
        
        '''Finding the RGB values of colormap representatives after median cut
           Input : Flattened image array
        '''
        # averaging the RGB values of all pixels in the box
        r_average = np.mean(img_arr[:,0])
        g_average = np.mean(img_arr[:,1])
        b_average = np.mean(img_arr[:,2])
        
        #storing the representative color
        colormap.append([r_average,g_average,b_average])
    
    def bucket_split(img_arr, depth):
        
        ''' Splitting the pixels into buckets using median cut algorithm recursively
            Input : img_arr - Flattened image array, depth - Depth of bucket ( K(number of represenatatives in colormap) = 2^depth)
        '''
        
        #base condition
        if len(img_arr) == 0:
            return 

        #depth of bucket algorithm    
        if depth == 0:
            K_representative(img_arr)
            return
        
        #calculating range of each color channel
        range_list = []
        range_list.append(np.max(img_arr[:,0]) - np.min(img_arr[:,0]))
        range_list.append(np.max(img_arr[:,1]) - np.min(img_arr[:,1]))
        range_list.append(np.max(img_arr[:,2]) - np.min(img_arr[:,2]))
 
        #calculating the channel ith max range
        max_range_channel = range_list.index(max(range_list))

        # sort the image pixels by color space with max range 
        # and find the median
        img_arr = img_arr[img_arr[:,max_range_channel].argsort()]
        median_index = int((len(img_arr)+1)/2)

        
        #split the array into two blocks using median
        bucket_split(img_arr[0:median_index], depth-1)
        bucket_split(img_arr[median_index:], depth-1)
 
    num_K = 2
    bucket_split(flattened_img_array,  math.log(num_K,2))

    #converting colormap to numpy array
    colormap = np.array(colormap)

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

    cv2.imwrite('result_median_cut.png',cv2.cvtColor(quantized_image, cv2.COLOR_RGB2BGR))

    #displaying quantized image
    cv2.imshow('Median cut quantized image',cv2.cvtColor(quantized_image, cv2.COLOR_RGB2BGR))

    print("Press any key to exit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == '__main__':
    
    main()





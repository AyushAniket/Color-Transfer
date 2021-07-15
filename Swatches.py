#importing libraries
import cv2
import numpy as np
import random
import math
from scipy.spatial import KDTree


def main():

    """Implementation of color transfer using swatches"""
   
    #reading source and target images
    image1 = cv2.imread('color.jpg')  
    image2 = cv2.imread('gray.png')

    #converting the images to LAB color space and data type float32 for numerical operations
    source = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB).astype("float32")

    #specifying number of swatches to input
    num_swatches = 2

    #list for storing all the swatches array
    colored_swatches = []

    while(num_swatches !=0):
    
        #defining function to be executed upon mouse click on the source image   
        def click_event1(event, x, y, flags, params): 
          
            # checking for left mouse clicks 
            if event == cv2.EVENT_LBUTTONDOWN:

                #saving the coordinates of the swatches
                coords.append([x,y])

                #displaying the points clicked
                cv2.circle(image1,(x,y),1,(255,255,255), 2) 
                cv2.imshow('source', image1)

                if len(coords)%2 == 0:
                    cv2.waitKey(500)
                    cv2.destroyAllWindows()

        #defining function to be executed upon mouse click on the target image 
        def click_event2(event, x, y, flags, params): 
          
            # checking for left mouse clicks 
            if event == cv2.EVENT_LBUTTONDOWN:

                #saving the coordinates of the swatches
                coords.append([x,y])

                #displaying the points clicked
                cv2.circle(image2,(x,y),1,(255,255,255), 2) 
                cv2.imshow('target', image2)

                if len(coords)%2 == 0:
                    cv2.waitKey(500)
                    cv2.destroyAllWindows()

        #list to store coordinates of the swatches
        coords = []

        print("Please select the diagonal points for source swatch")
        
        #acquiring source swatch
        cv2.imshow('source',image1) 
        cv2.setMouseCallback('source', click_event1)  # setting mouse handler for the image 
                                                      # and calling the click_event() function 
        cv2.waitKey(0) # wait for a key to be pressed to exit 
        cv2.destroyAllWindows()# close the window 

        print("Please select the corresponding diagonal points for target swatch")

        #acquiring target swatch
        cv2.imshow('target',image2) 
        cv2.setMouseCallback('target', click_event2) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

        #source and target swatch array from coordinates
        source_swatch = source[coords[0][1]:coords[1][1]+1,coords[0][0]:coords[1][0]+1,:]
        target_swatch = target[coords[2][1]:coords[3][1]+1,coords[2][0]:coords[3][0]+1,:]

        #random jittered sapmling 50 pixels from source and target swatch array
        sampled_source_swatch = random_sampling(source_swatch,50)
        sampled_target_swatch = random_sampling(target_swatch,50)

        
        #performing luminance remapping on the source image swatch
        sampled_source_swatch[:,:,0] = luminiance_remap(sampled_source_swatch[:,:,0],sampled_target_swatch[:,:,0])
        sampled_source_swatch[:,:,0] = np.clip(sampled_source_swatch[:,:,0], 0, 255)

        #computing standard deviation for each pixel in a 5x5 neighbourhood
        source_std = sd_neighbourhood(sampled_source_swatch[:,:,0],5)
        target_std = sd_neighbourhood(sampled_target_swatch[:,:,0],5)

        #transfer of colors to target image swatch
        swatch_transfered = color_transfer(sampled_source_swatch,sampled_target_swatch,source_std,target_std)

        
        #storing colored target swatch
        colored_swatches.append(swatch_transfered)

        num_swatches -= 1

    


    #texture synthesis algorithm color transfer from colored swatch to target image
    target = texture_synthesis(target,colored_swatches)
    final_image = cv2.cvtColor(target.astype("uint8"),cv2.COLOR_LAB2BGR)

    cv2.imwrite('result_swatches.png',final_image)


    #displaying final image
    cv2.imshow('final_image',final_image)
    print("Press any key to exit")
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
     


#required functions

def texture_synthesis(target,colored_swatches):
    
    """ texture synthesis algorithm color transfer from colored swatch to target image
        with a neighbourhood size of 3x3
        Input - target - target omage array, colored_swatches - colored target swatches array
        Output: target - colored target image array
    """

    #length and breadth of target image and colred swatches
    y_t,x_t = target[:,:,0].shape
    y_sw1,x_sw1 = colored_swatches[0][:,:,0].shape
    y_sw2,x_sw2 = colored_swatches[1][:,:,0].shape
  

    #padding the image and swatches for boundary pixels
    padded_target = np.pad(target[:,:,0], (1,1))
    padded_swatch1 = np.pad(colored_swatches[0][:,:,0], (1,1))
    padded_swatch2 = np.pad(colored_swatches[1][:,:,0], (1,1))
  

    #flattening colored swatches
    array1 = padded_swatch1[0:3,0:3].reshape([1,9])
    array2 = padded_swatch2[0:3,0:3].reshape([1,9])

    for y in range(1,y_sw1+1):
        for x in range(1,x_sw1+1):

            if y == 1 and x==1:
                continue
            else:

                flat  = padded_swatch1[y-1:y+2,x-1:x+2].reshape([1,9])
                array1 = np.hstack((array1,flat))

    for y in range(1,y_sw2+1):
        for x in range(1,x_sw2+1):

            if y == 1 and x==1:
                continue
            else:

                flat  = padded_swatch2[y-1:y+2,x-1:x+2].reshape([1,9])
                array2 = np.hstack((array2,flat))

    array1 = array1.reshape([y_sw1*x_sw1,9])
    array2 = array2.reshape([y_sw2*x_sw2,9])

    #flattened colored swatches array with 3x3 window size neighbourhood
    array = np.append(array1,array2,axis = 0)

    #Using K_D tree Algorithm for matching texture
    tree = KDTree(array)

    #finding texture match pixel by pixel 
    for i in range(1,y_t+1):
        for j in range(1,x_t+1):

            flat  = padded_target[i-1:i+2,j-1:j+2].reshape([1,9])
            _,ii = tree.query(flat,k=1)

            if ii < (y_sw1*x_sw1):

                y_index = int(math.floor(int(ii)/x_sw1))
                x_index = int(ii) - x_sw1*y_index
                              
                target[i-1,j-1,1] = colored_swatches[0][y_index,x_index,1]
                target[i-1,j-1,2] = colored_swatches[0][y_index,x_index,2]

            else:

                ii = int(ii) - y_sw1*x_sw1

                y_index = int(math.floor(ii/x_sw2))
                x_index = ii - x_sw2*y_index

                target[i-1,j-1,1] = colored_swatches[1][y_index,x_index,1]
                target[i-1,j-1,2] = colored_swatches[1][y_index,x_index,2]
                
    return target    

def random_sampling(source_image,number_of_samples):

    '''Collecting random jittered samples by grid mapping and taking single random pixel from each cell
       Input: source_image - saource image array, number_of_samples - number of samples required (around 50)
       Output: sampled - around 50 sample pixels of nxn shape
    '''
    
    #size of nxn grid
    n = int(np.sqrt(number_of_samples))

    #height and breadth of source image
    y, x = source_image[:,:,0].shape

    #y-axis and x-axis step size for grid
    step_y = y // n
    step_x = x // n
    
    sampled = np.zeros([n,n,3])
    
    for i in range(0,n):
        for j in range(0,n):
            
            actual_i = random.randrange(i*step_y, (i+1)*step_y-1, 1)
            actual_j = random.randrange(j*step_x, (j+1)*step_x-1, 1)
            
            sampled[i,j,:] = source_image[actual_i,actual_j,:]
            
    return sampled

def color_transfer(source,target,source_std,target_std):

    '''transfer of colors to greyscale image
       Input : source - source image array, target - target image array
               source_std - standard deviation of source sampled pixels
               target_std - standard deviation of target pixels

       Output : transformed colored greysacle image
    '''

    #splitting source and target image into LAB color channels
    (l_t,a_t,b_t) = cv2.split(target)
    (l_s,a_s,b_s) = cv2.split(source)
    
    y, x = l_t.shape
    
    for i in range(y):
        for j in range(x):
            
            weighted_sum = 0.5 * np.square(l_s - l_t[i,j]) + 0.5 * np.square(source_std - target_std[i,j])

            #finding the pixel with minimum weighted sum and transferring a b pixel values
            index = np.argwhere(weighted_sum == np.min(weighted_sum))
        
            a_t[i,j] = a_s[index[0,0],index[0,1]]
            b_t[i,j] = b_s[index[0,0],index[0,1]]
            
    transformed = cv2.merge([l_t,a_t,b_t])
    transformed = transformed.astype("uint8")
    
    return transformed

def sd_neighbourhood(l_channel, neighbourhood_size):

    '''computing standard deviation for each pixel in a 5x5 neighbourhood
       Input : l_s - source/target image l channel, neighbourhood_size - window size (5)
       Output : sds - standard deviation of l channel
    '''
    
    amt_to_pad = (neighbourhood_size - 1) // 2

    y, x = l_channel.shape
    sds = np.zeros(l_channel.shape)

    #padding the image for boundary pixels
    padded = np.pad(l_channel, (amt_to_pad, amt_to_pad))
    
    for i in range(amt_to_pad+1,y+2):
        for j in range(amt_to_pad+1,x+2):
            region = padded[i-2:i+2, j-2:j+2]
            sd = np.std(region[:])
            sds[i-2, j-2] = sd
            
    return sds



def luminiance_remap(l_s, l_t):

    '''luminance remapping on the source image
       Input : l_s - source image l channel, l_t - target image l channel
       Output : remapped - remapped l_s channel
    '''
     
    mu_s = np.mean(l_s)
    mu_t = np.mean(l_t)
    sigma_s = np.std(l_s)
    sigma_t = np.std(l_t)
    remapped = sigma_t / sigma_s * (l_s - mu_s) + mu_t
    
    return remapped





if __name__ == '__main__':
    
    main()

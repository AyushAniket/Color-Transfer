import cv2
from  matplotlib import pyplot as plt
import numpy as np
import random


def main():

    '''Color transfer to greyscale image using golbal matching procedure and
       random jittered sampling
    '''

    #reading source and target images
    image1 = cv2.imread('color.jpg')  
    image2 = cv2.imread('gray.jpg')

    #displaying images
    cv2.imshow('image',image1)
    print("Press any key to continue")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('image',image2)
    print("Press any key to continue")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    #converting the images to LAB color space and data type float32 for numerical operations
    source = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB).astype("float32")

    #performing luminance remapping on the source image
    source[:,:,0] = luminiance_remap(source[:,:,0],target[:,:,0])
    source[:,:,0] = np.clip(source[:,:,0], 0, 255)

    #taking random jittered samples 
    sampled_source = random_sampling(source,200)

    #computing standard deviation for each pixel in a 5x5 neighbourhood
    source_std = sd_neighbourhood(sampled_source[:,:,0],5)
    target_std = sd_neighbourhood(target[:,:,0],5)

    #transfer of colors to greyscale image
    final_image = color_transfer(sampled_source,target,source_std,target_std)

    cv2.imwrite('result_global.png',final_image)

    #displaying images
    cv2.imshow('image',final_image)
    print("Press any key to exit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def random_sampling(source_image,number_of_samples):

    '''Collecting random jittered samples by grid mapping and taking single random pixel from each cell
       Input: source_image - saource image array, number_of_samples - number of samples required (around 200)
       Output: sampled - around 200 sample pixels of nxn shape
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
    transformed = cv2.cvtColor(transformed.astype("uint8"),cv2.COLOR_LAB2BGR)
    
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
    remapped = (l_s - mu_s)* (sigma_t / sigma_s) + mu_t
    
    return remapped



if __name__ == '__main__':

    main()

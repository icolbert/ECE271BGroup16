import cv2
import numpy as np
import csv

def rescale_segment( segment, size = [28,28], pad = 0 ):
    '''function for resizing (scaling down) images
    input parameters
    seg : the segment of image (np.array)
    size : out size (list of two integers)
    output 
    scaled down image'''
    if len(segment.shape) == 3 : # Non Binary Image
        # thresholding the image
        ret,segment = cv2.threshold(segment,127,255,cv2.THRESH_BINARY)
    m,n = segment.shape
    idx1 = list(range(0,m, (m)//(size[0]) ) )
    idx2 = list(range(0,n, n//(size[1]) )) 
    out = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            out[i,j] = segment[ idx1[i] + (m%size[0])//2, idx2[j] + (n%size[0])//2]
#     if pad
    return out

def extract_segments(img, pad=30, reshape = 0,size = [28,28]) :
    '''function to extract individual chacters and digits from an image
    input paramterts
    img : input image (numpy array)
    pad : padding window size around segments (int)
    size : out size (list of two integers)
    reshape : if 1 , output will be scaled down. if 0, no scaling down
    Returns
    out : list of each segments (starting from leftmost digit)'''
    # thresholding the image
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    #print(thresh1)
    # Negative tranform gray levels (background becomes black) 
    thresh1 = thresh1 - 1
    thresh1[thresh1 == 254] = 0
    thresh1[thresh1 == - 1 ] = 255
    
    # connected component labelling 
    output = cv2.connectedComponentsWithStats(thresh1, 4)
    final = []
    temp1 = np.sort( output[2][:,0] )
    kernel = np.ones( [3,3])
    for i in range(1,output[0]):
        temp2 = output[2]
        cord = np.squeeze( temp2[temp2[:,0] == temp1[i]] )
        num = np.pad( thresh1[ cord[1]:cord[1]+cord[3], cord[0]:cord[0]+cord[2] ], pad,'constant')
        num = cv2.dilate(num,kernel,iterations = 1)
        if reshape == 1:
            num = rescale_segment( num, size )
        final.append(num)        
    return final

def collate_symbols(symbol_id,sym_filename,img_filename): 
    """function strive to create dataset from HAsy by scanning given csv file for desired symbol(symbol_id)
    parameters:
    symbol_id:(type integer) id of symbol to be saved in csv
    sym_filename:(type string) filename of csv file containing the symbols
    img_filename:(type string) filename of csv file containing the matrix values of the images"""
    latex_val = [train1["latex"][train1["symbol_id"]==symbol_id].iloc[0]]
    #print(type(latex_val));
    folder_loc = 'C:/Users/pmath/Documents/271B project/'
    for i,file_loc in enumerate(train1["path"][train1["symbol_id"]==symbol_id]):
        img1 = cv2.imread(folder_loc + file_loc,0);
        img1 = (255 -img1)/255;
        img1 = np.pad(img1,12,'constant',constant_values=(0,0));
        kernel = np.ones((2,2),np.uint8);
        img1 = cv2.dilate(img1,kernel,iterations = 1)
        img1 = rescale_segment(img1,[28,28]);
        row = img1.reshape(1,-1).tolist();
        row = row[0]; 
        with open(img_filename,"a",newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(row);
        with open(sym_filename,"a",newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(latex_val);
            
def collate_numbers(images,labels,num_img_filename,num_sym_filename):
    """function creates csv files (one for images and another for the labels provided).
    parameters:
    images:(type list)list of images loaded """
    train_images = np.array(images)
    train_images = (train_images)/255
    train_labels = np.array(labels)
        
    num_imgs = train_images.shape[0]

    permutation = list(np.random.permutation(num_imgs))
    shuffled_x = train_images[permutation,:]
    shuffled_t = train_labels[permutation]
    
    batch = zip(shuffled_x,shuffled_t)
    for i,(x,t) in enumerate(batch):
        if(i>20000):
            break
        with open(num_img_filename,"a",newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(x.tolist());   
        with open(num_sym_filename,"a",newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow([t]);

if __name__ == '__main__':
	from matplotlib import pyplot as plt

	img = cv2.imread('/Users/josejoy/Desktop/ECE 271B Stat Learning /project/eq1.jpg',0)
	segments= extract_segments(img, 10)
	plt.imshow(segments[1])
	plt.show()


    

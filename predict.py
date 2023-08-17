import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from pathlib import Path

import models

    
# Default input size
height = 1080#228
width = 1920#304
channels = 3
batch_size = 1

# Create a placeholder for the input image
input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

# Construct the network
net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        

def predict(model_data_path, image_path):   
    # Read image
    img = Image.open(image_path)
    #img = img.resize([width,height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)

    with tf.Session() as sess:
        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        #net.load(model_data_path, sess) 

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        
        # Plot result
        #fig = plt.figure()
        #ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        #fig.colorbar(ii)
        # plt.savefig('final.jpg')
        # image = cv2.imread('final.jpg')
        # graysc = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('finalgray.jpg',graysc)

        #New: Get only image
        filename = Path(image_path).stem
        #print(filename)
        depth_path_image = f"img/{filename}_depth.jpg"
        depth_path_image_color = f"img-depth-color/{filename}_depth_color.jpg"
        plt.imsave(depth_path_image_color, pred[0,:,:,0])
        image = cv2.imread(depth_path_image_color, cv2.IMREAD_UNCHANGED)
        dim = (width, height)
        imagerz = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        graysc = cv2.cvtColor(imagerz, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(depth_path_image,graysc)

    return pred
        
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    directory = 'img-raw/'
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            #print(filename)
            #print(filepath)

            # Predict the image
            pred = predict(args.model_path, filepath)
    
    os._exit(0)

if __name__ == '__main__':
    main()

        




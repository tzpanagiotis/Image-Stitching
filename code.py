
#Program that stiches 2 or 3 images together.
#Loading libraries.
import numpy as np
import cv2
import warnings

#Ask the user to put 2 or 3 images as input, each one seperated by comma.
inp = input("Type 2 or 3 image files separated by commas (in order from left to right): ")
img_files = inp.split(sep=',')

#load the images the user just gave.
print("\nLoading images, resizing and grayscale...\n")
my_imgs = [cv2.imread(file) for file in img_files]

#resize the images.
for i in range(len(my_imgs)):
    if my_imgs[i].shape[1] > 2500 or my_imgs[i].shape[0] > 2500:
        width = int(my_imgs[i].shape[1]*0.15)
        height = int(my_imgs[i].shape[0]*0.15)
        dim = (width,height)
        my_imgs[i]=cv2.resize(my_imgs[i],dim,interpolation=cv2.INTER_AREA)

#The images are displayed in windows.
for i in range(len(my_imgs)):
    cv2.imshow('Image %d'%(i+1),my_imgs[i])
cv2.waitKey(0)

#Convert the images from the initial color space to grayscale.
my_imgs_gray = [cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for img in my_imgs]

#The stitching has to happen N-1 times, where N the number of the images.
for step in range(len(img_files)-1):
    print("*"*8 + " STITCHING PART %d OF %d "%(step+1,len(img_files)-1) + "*"*8)

    #For the first step, you only need the first two images.
    if step == 0:
        imgs = my_imgs[:2]
        imgs_gray = my_imgs_gray[:2]
    #If tere is a second step, the third image is stitched to the result of the previous step.
    elif step == 1:
        imgs = [dst,my_imgs[2]]
        imgs_gray = [cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY),my_imgs_gray[2]]

    #Keypoints are extracted and descriptors are computed using the Scale Invariant Feature Transform (SIFT) algorithm.
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(imgs_gray[0],None)
    kp2, des2 = sift.detectAndCompute(imgs_gray[1],None)
    #The keypoints are drawn in each picture, and then the results are displayed in windows.
    cv2.imwrite('left_keypoints_step_%d_of_%d.jpg'%(step+1,len(img_files)-1),cv2.drawKeypoints(imgs_gray[0],kp1,None))
    cv2.imwrite('right_keypoints_step_%d_of_%d.jpg'%(step+1,len(img_files)-1),cv2.drawKeypoints(imgs_gray[1],kp2,None))
    cv2.imshow('Left Image Keypoints (Step %d of %d)'%(step+1,len(img_files)-1),cv2.drawKeypoints(imgs_gray[0],kp1,None))
    cv2.imshow('Right Image Keypoints (Step %d of %d)'%(step+1,len(img_files)-1),cv2.drawKeypoints(imgs_gray[1],kp2,None))
    cv2.waitKey(0)

    #Create BFMatcher object.
    match = cv2.BFMatcher()
    #Match descriptors.
    matches = match.knnMatch(des1,des2,k=2)
    #Out of all the matches, the ones that fulfill the condition are only picked.
    selected = []
    for m,n in matches:
        if m.distance < n.distance/3:
            selected.append(m)
    print("Selected matches (Step %d of %d): %d"%(step+1,len(img_files)-1,len(selected)))

    draw_params = dict(matchColor=(0,255,0),
                        singlePointColor=None,
                        flags=2)

    #Draw the matches in the images and then display them in windows.
    img_match = cv2.drawMatches(imgs[0],kp1,imgs[1],kp2,selected,None,**draw_params)
    cv2.imwrite('features_matched_step_%d_of_%d.jpg'%(step+1,len(img_files)-1),img_match)
    cv2.imshow('Matches (Step %d of %d)'%(step+1,len(img_files)-1),img_match)
    cv2.waitKey(0)

    #If the matches are less then 4, they are not enough to stitch the images.
    if len(selected) < 4:
        print("Not enough matches are found - %d/4"%(len(selected)))
        exit(1)

    #If enough matches are found, the locations of matched keypoints are extracted in both the images. They are passed to find the Homography matrix using the RANSAC algorithm.
    src_pts = np.float32([ kp2[m.trainIdx].pt for m in selected ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp1[m.queryIdx].pt for m in selected ]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    dst = cv2.warpPerspective(imgs[1],H,(imgs[0].shape[1] + imgs[1].shape[1], imgs[0].shape[0]))

    print('Homography Array (Step %d of %d):'%(step+1,len(img_files)-1))
    print(H)

    print('Generating the stitched image (Step %d of %d)...\n'%(step+1,len(img_files)-1))
    #Blend the images.
    for i in range(imgs[0].shape[0]):
        overlap_start = 0
        while np.array_equal(dst[i][overlap_start],np.array([0,0,0])) and overlap_start < imgs[0].shape[1]:
            dst[i][overlap_start] = imgs[0][i][overlap_start]
            overlap_start += 1

        if overlap_start < imgs[0].shape[1] -1:
            dst[i][overlap_start] = imgs[0][i][overlap_start]
            dst[i][overlap_start+1] = imgs[0][i][overlap_start+1]

        for j in range(overlap_start+2,imgs[0].shape[1]):
            coef = (j - overlap_start)/(imgs[0].shape[1] - overlap_start)
            dst[i][j] = (1-coef)*dst[i][j] + coef*imgs[0][i][j]

    #display the result in a window.
    cv2.imshow("Stitched Image (Step %d of %d)"%(step+1,len(img_files)-1), dst)
    cv2.imwrite('stitched_step_%d_of_%d.jpg'%(step+1,len(img_files)-1),dst)
    cv2.waitKey(0)

    #Crop right part of stitched image if a second step follows.
    if(len(img_files) == 3):
        crop_lim = dst.shape[1]
        for i in range(dst.shape[0]):
            col = dst.shape[1] - 1
            while(np.array_equal(dst[i][col],np.array([0,0,0]))):
                col -= 1
            if col < crop_lim:
                crop_lim = col
        dst = dst[:,:crop_lim]

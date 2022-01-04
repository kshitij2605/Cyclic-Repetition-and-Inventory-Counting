import cv2
import numpy as np
import time
import random
import os
import skvideo.io

postfix = 1

# EDIT
Dataset_Path = "Data"
Output_Path = "Augmented_Data_test6"
# EDIT


'''
Same Video
'''
def A00(video,s) :

    fps = int(video.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    while ret:
        result.write(src) 
        ret,src = video.read() 

    result.release() 


'''
Mirror
'''
def A01(video,s) :

    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   
    
    ret,src = video.read()

    while ret:
        # EDIT
        src = src[:, ::-1, :]
        result.write(src) 
        # EDIT
        ret,src = video.read() 
    
    result.release()


'''
Invert
'''
def A02(video,s) : 

    fps = int(video.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    while ret:
        # EDIT
        src = src[::-1, :, :]
        result.write(src) 
        # EDIT
        ret,src = video.read() 
    
    result.release()


'''
Mirror + Invert
'''
def A03(video,s) : 

    fps = int(video.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    while ret:
        # EDIT
        src = src[::-1, ::-1, :]
        result.write(src) 
        # EDIT
        ret,src = video.read()  

    result.release()


'''
Camera Shaking
'''
def A11(video,s) :

    # EDIT
    ct = 0
    ct_add = True
    angle_change = 1
    angle = 0
    neg = False
    pos = False
    # EDIT

    fps = int(video.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    while ret:

        # EDIT
        center = (src.shape[1]//2, src.shape[0]//2)
        angle_change = random.uniform(-5, 5)
        if(angle>45) : neg = True
        if(angle<-45) : pos = True
        if(neg) : 
            angle_change = random.uniform(-5,0)
            if(angle<20) : neg = False
        if(pos) : 
            angle_change = random.uniform(0,5)
            if(angle>-20) : pos = False
        angle = angle + angle_change
        scale = 1
        rot_mat = cv2.getRotationMatrix2D( center, angle, scale )
        warp_rotate_dst = cv2.warpAffine(src, rot_mat, (src.shape[1], src.shape[0]))
        result.write(warp_rotate_dst) 
        # EDIT
        ret,src = video.read() 

    result.release()


'''
Camera Shaking + Scale Changing
'''
def A12(video,s) : 

    # EDIT
    ct = 0
    ct_add = True
    angle_change = 1
    angle = 0
    scale = 1
    scale_change = 0
    neg = False
    pos = False
    # EDIT

    fps = int(video.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    while ret:

        # EDIT
        center = (src.shape[1]//2, src.shape[0]//2)
        angle_change = random.uniform(-5, 5)
        if(angle>45) : neg = True
        if(angle<-45) : pos = True
        if(neg) : 
            angle_change = random.uniform(-5,0)
            if(angle<20) : neg = False
        if(pos) : 
            angle_change = random.uniform(0,5)
            if(angle>-20) : pos = False
        angle = angle + angle_change
        scale_low = 0.6
        scale_up = 1
        scale_change = random.uniform(-0.1,0.1)
        if(scale<(scale_low+0.1)) : scale_change = random.uniform(0,0.1)
        if(scale>(scale_up-0.1)) : scale_change = random.uniform(-0.1,0)
        scale = scale + scale_change
        if ct_add : ct = ct + 1
        else : ct = ct - 1
        rot_mat = cv2.getRotationMatrix2D( center, angle, scale )
        warp_rotate_dst = cv2.warpAffine(src, rot_mat, (src.shape[1], src.shape[0]))
        result.write(warp_rotate_dst) 
        # EDIT
        ret,src = video.read() 

    result.release()


'''
Continuous Rotation
'''
def A13(video,s) :

    # EDIT
    ct = 0
    ct_add = True
    angle_change = 1
    angle = 0
    # EDIT

    fps = int(video.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    while ret:

        # EDIT
        center = (src.shape[1]//2, src.shape[0]//2)
        angle_change = random.uniform(0,1)
        angle = angle + angle_change
        scale = 1
        rot_mat = cv2.getRotationMatrix2D( center, angle, scale )
        warp_rotate_dst = cv2.warpAffine(src, rot_mat, (src.shape[1], src.shape[0]))
        result.write(warp_rotate_dst)
        # EDIT 
        ret,src = video.read() 

    result.release()


'''
Warping 
'''
def A21(video,s) :

    # EDIT
    ct = 0
    ct_add = True
    angle_change = 1
    angle = 0
    c1 = 0.33
    c2 = 0.85
    c3 = 0.25
    c4 = 0.15
    c5 = 0.7
    # EDIT

    fps = int(video.get(cv2.CAP_PROP_FPS))

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    while ret:

        # EDIT
        srcTri = np.array( [[0, 0], [src.shape[1] - 1, 0], [0, src.shape[0] - 1]] ).astype(np.float32)
        v1 = random.uniform(-0.1,0.1)
        v2 = random.uniform(-0.1,0.1)
        v3 = random.uniform(-0.1,0.1)
        v4 = random.uniform(-0.1,0.1)
        v5 = random.uniform(-0.1,0.1)
        if(c1>0.4) : c1 = c1-abs(v1)
        elif(c1<0.1) : c1 = c1 + abs(v1)
        else : c1 = c1 + v1
        if(c2>0.9) : c2 = c2-abs(v2)
        elif(c2<0.7) : c2 = c2 + abs(v2)
        else : c2 = c2 + v2
        if(c3>0.9) : c3 = c3-abs(v3)
        elif(c3<0.5) : c3 = c3 + abs(v3)
        else : c3 = c3 + v3
        if(c4>0.9) : c4 = c4-abs(v4)
        elif(c4<0.5) : c4 = c4 + abs(v4)
        else : c4 = c4 + v4
        if(c5>0.9) : c5 = c5-abs(v5)
        elif(c5<0.9) : c5 = c5 + abs(v5)
        else : c5 = c5 + v5
        dstTri = np.array( [[0, src.shape[1]*c1], [src.shape[1]*c2, src.shape[0]*0.25], [src.shape[1]*0.15, src.shape[0]*c5]] ).astype(np.float32)
        warp_mat = cv2.getAffineTransform(srcTri, dstTri)
        warp_dst = cv2.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))
        result.write(warp_dst) 
        # EDIT
        ret,src = video.read() 

    result.release()


'''
fast forwarded
'''
def A23(video,s) :

    fps = int(video.get(cv2.CAP_PROP_FPS)) * 4

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    while ret:
        result.write(src) 
        ret,src = video.read() 

    result.release() 



'''
Slowed
'''
def A24(video,s) :

    fps = int(video.get(cv2.CAP_PROP_FPS)) / 4

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    while ret:
        result.write(src) 
        ret,src = video.read() 

    result.release()


'''
Skipping n frames (sub-sampling)
'''
def A25(video,s):
    count = 0
    fps = int(video.get(cv2.CAP_PROP_FPS)) / 4

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    while video.isOpened():
        ret, src = video.read()

        if ret:
            # cv2.imwrite('frame{:d}.jpg'.format(count), frame)
            count += 10 # i.e. at 10 fps, this advances one third of a second
            video.set(cv2.CAP_PROP_POS_FRAMES, count)
            result.write(src)
            # cv2.imshow('frame', src)
        else:
            # cap.release()
            break

    #     cv2_imshow('output.avi')

    # release VideoCapture()
    result.release()



'''
Thresholding
'''
def A26(cap,s) :
    fps = int(video.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    
    # gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
    # cv2.imshow('window', frame1)
    ret1, frame1 = cap.read()
    ret2 = ret1
    while ret2:
        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()
        # gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2)
        # gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
        if ret2:
            deltaframe = cv2.absdiff(frame2, frame1)
            # cv2.imshow('delta', deltaframe)
            threshold = cv2.threshold(deltaframe, 25, 255, cv2.THRESH_BINARY)[1]
            threshold = cv2.dilate(threshold, None)
            # cv2.imshow('threshold', threshold)
            out.write(threshold)
            countour, heirarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i in countour:
                if cv2.contourArea(i) < 50:
                    continue

                (x, y, w, h) = cv2.boundingRect(i)
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # cv2.imshow('window', frame2)

        # if cv2.waitKey(27) == ord('q'):
        #     break
        else:
            break

    out.release()


'''
Gaussian blur
'''
def P00(video,s):
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    fps = int(video.get(cv2.CAP_PROP_FPS)) / 4

    result = cv2.VideoWriter(f'{Output_Path}\\{s[:-4]}_{postfix}.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(int(video.get(3)),int(video.get(4))))   

    ret,src = video.read()

    while (video.isOpened()):
        # capture each frame of the video
        ret, src = video.read()
        if ret == True:
            # add gaussian blurring to frame
            src = cv2.GaussianBlur(src, (5, 5), 0)
            # save video frame
            result.write(src)
            # display frame
            # cv2_imshow('video', frame)
            # show_video('/tmp/video.mp4')
            # cv2.imshow('frame',frame)
            # press `s` to exit
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        # if no frame found
        else:
            break

    #     cv2_imshow('output.avi')

    # release VideoCapture()
    result.release()

#========================================================================================

# EDIT
Augment = [A00,A01,A02,A11,A12,A13,A21,A23,A25,A26,P00]
# EDIT

coun = 0

for file in os.listdir(Dataset_Path) :

    try : 

        postfix = 1

        path=os.path.join(f"{Dataset_Path}\\", file)
        s = str(file)

        for aug in Augment :
            video = cv2.VideoCapture(path)
            aug(video,s)
            postfix = postfix+1
        
        coun = coun + 1
        print(f"Videos Augmented : {coun}")

    except : pass

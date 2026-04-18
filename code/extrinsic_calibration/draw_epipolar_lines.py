import cv2 as cv
import numpy as np

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r = img1.shape[0]
    c = img1.shape[1]
    
    for r,pt1,pt2 in zip(lines,pts1,pts2):

        print(f"Pt1: {(pts1[0][0], pts1[1][0])}")
        print(f"Pt2: {(pts2[0][0], pts2[1][0])}")
        
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,(int(pts1[0][0]), int(pts1[1][0])),5,color,-1)
        img2 = cv.circle(img2,(int(pts2[0][0]), int(pts2[1][0])),5,color,-1)
    return img1,img2


def drawEpipolarLines(F, img1, img2, pts1, pts2, counter):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
    cv.imwrite(f"./epipolar_lines/cam746/image{counter}.png", img5)
    cv.imwrite(f"./epipolar_lines/cam746/image{counter+1}.png", img6)
    
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
    cv.imwrite(f"./epipolar_lines/cam745/image{counter}.png", img3)
    cv.imwrite(f"./epipolar_lines/cam745/image{counter+1}.png", img4)
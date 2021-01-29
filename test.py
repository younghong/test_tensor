import cv2 as cv
import sys
import json
import os
from PIL import Image
import pytesseract

def im_trim (idx, x , y, w, h, img): #함수로 만든다
    img_trim = img[y:y+h, x:x+w] #trim한 결과를 img_trim에 담는다
    imgName = 'crop_' + str(idx) +'.jpg'
    cv.imwrite(imgName,img_trim) #org_trim.jpg 라는 이름으로 저장
    return img_trim #필요에 따라 결과물을 리턴

def main(string):
    print(string)
    org_image = cv.imread(string)
    img_color = cv.imread(string, cv.IMREAD_COLOR)
    cv.imshow('result', img_color)
    cv.waitKey(0)

    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    cv.imshow('result', img_gray)
    cv.waitKey(0)

    ret, img_binary = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow('result', img_binary)
    cv.waitKey(0)

    contours, hierarchy = cv.findContours(img_binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    items=[]

    for cnt in contours:
        size = len(cnt)
        print(size)

        epsilon = 0.005 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)

        size = len(approx)
        print(size)

        cv.line(img_color, tuple(approx[0][0]), tuple(approx[size - 1][0]), (0, 255, 0), 3)
        for k in range(size - 1):
            cv.line(img_color, tuple(approx[k][0]), tuple(approx[k + 1][0]), (0, 255, 0), 3)

        if cv.isContourConvex(approx):
            if size == 4:
                x, y, width, height = cv.boundingRect(cnt)
                trim_image = im_trim(cnt, x, y, width, height , org_image)  # trim_image 변수에 결과물을 넣는다
                cv.imshow('result', trim_image)
                cv.waitKey(0)
                #text = pytesseract.image_to_string(trim_image, lang='kor')
                text = pytesseract.image_to_string(trim_image, lang='kor', config='--psm 1 -c preserve_interword_spaces=1')
                print(text)
                items.append( {'x':x,'y':y,'w':width,'h':height, 'text':text} )
            else:
                print('not rect1: ')

        else:
            print('not rect2')

    jsonObject = {'items':items}
    jsonString = json.dumps(jsonObject)
    print(jsonString)
    f = open('D:/dev37/ubiformJson.txt', 'w')
    f.write(jsonString)
    f.close()

    cv.imshow('result', img_color)
    #cv.waitKey(0)
    os.system(r'"C:/Program Files (x86)/UBIFORM/UBIFORM Editor/UBIReport4Starter.exe" '+'OCR '+'D:/dev37/ubiformJson.txt')

if __name__ == "__main__":
    #cv.waitKey(0)
    main('D:/dev37/a.png')
    #main('D:/dev37/eform.png')

    #main(sys.argv[1])

import cv2
import numpy as np 
import pytesseract 
import time
from PIL import Image
from copy import copy

flag=0


def printToImg(board,to):
      for x in range (0,9):
            for y in range (0,9):
                  font = cv2.FONT_HERSHEY_SIMPLEX
                  if(printingBoard[x][y]==1):
                        #cv2.putText(to,"X",((50*y)+5,(50*x)+40), font, 1.5,(0,255,255),2,cv2.LINE_AA)
                        cv2.rectangle(to,((50*y)+5,(50*x)+5),((50*y)+35,(50*x)+45),(255,255,255),-1)
                        cv2.putText(to,str(board[x][y]),((50*y)+5,(50*x)+40), font, 1.5,(0,0,255),2,cv2.LINE_AA)   
                  elif(board[x][y] == 0):
                        cv2.rectangle(to,((50*y)+5,(50*x)+5),((50*y)+35,(50*x)+45),(255,255,255),-1)
                  else:
                        if(InitialBoard[x][y] == 0):
                              cv2.putText(to,str(board[x][y]),((50*y)+5,(50*x)+40), font, 1.5,(0,0,255),2,cv2.LINE_AA)   
                  

def rectify(h):
      h = h.reshape((4,2))
      hnew = np.zeros((4,2),dtype = np.float32)
      add = h.sum(1)
      hnew[0] = h[np.argmin(add)]
      hnew[2] = h[np.argmax(add)]
       
      diff = np.diff(h,axis = 1)
      hnew[1] = h[np.argmin(diff)]
      hnew[3] = h[np.argmax(diff)]

      return hnew

def printBoard(board):
    for x in range (0,9):
          if(x%3==0):
              print("\n")
          for y in range(0,9):
              if(y%3==0):
                  print "|",
              print(board[x][y]),
          print "|"
    

def isFull(board):
    for x in range(0, 9):
        for y in range (0, 9):
            if board[x][y] == 0:
                return False
    return True
    
def possibleEntries(board, i, j):
    
    possibilityArray = {}
    
    for x in range (1, 10):
        possibilityArray[x] = 0
    
    #For horizontal entries
    for y in range (0, 9):
        if not board[i][y] == 0: 
            possibilityArray[board[i][y]] = 1
     
    #For vertical entries
    for x in range (0, 9):
        if not board[x][j] == 0: 
            possibilityArray[board[x][j]] = 1
            
    #For squares of three x three
    k = 0
    l = 0
    if i >= 0 and i <= 2:
        k = 0
    elif i >= 3 and i <= 5:
        k = 3
    else:
        k = 6
    if j >= 0 and j <= 2:
        l = 0
    elif j >= 3 and j <= 5:
        l = 3
    else:
        l = 6
    for x in range (k, k + 3):
        for y in range (l, l + 3):
            if not board[x][y] == 0:
                possibilityArray[board[x][y]] = 1          
    
    for x in range (1, 10):
        if possibilityArray[x] == 0:
            possibilityArray[x] = x
        else:
            possibilityArray[x] = 0
    
    return possibilityArray

def sudokuSolver(board,printingBoard):
    
    i = 0
    j = 0
    global flag
    
    possiblities = {}
    
    # if board is full, there is no need to solve it any further
    if (isFull(board)):
        print("Board Solved Successfully!")
        flag = 1
        printBoard(board)
        return
    
    else:
        # find the first vacant spot
        for x in range (0, 9):
            for y in range (0, 9):
                if board[x][y] == 0:
                    i = x
                    j = y
                    break
            else:
                continue
            break
        
        # get all the possibilities for i,j
        possiblities = possibleEntries(board, i, j)
        
        # go through all the possibilities and call the the function
        # again and again
        for x in range (1, 10):
            if not possiblities[x] == 0:
                board[i][j] = possiblities[x]
                printingBoard[i][j] = 1
                #file.write(printFileBoard(board))
                printToImg(board,puzzle)
                if(flag==0):
                      cv2.imshow('solver', puzzle)
                      cv2.waitKey(1)
                sudokuSolver(board,printingBoard)
        # backtrack
        board[i][j] = 0
        printingBoard[i][j] = 0

cap = cv2.VideoCapture(0)
time.sleep(3)

while(1):
      ret, img = cap.read()
      height, width, _ = img.shape
      #img = cv2.imread('sudoku.jpg')
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
      hierarchy,contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      biggest = None
      max_area = 0
      for i in contours:
          area = cv2.contourArea(i)
          if area > 100:
                  peri = cv2.arcLength(i,True)
                  approx = cv2.approxPolyDP(i,0.02*peri,True)
                  if area > max_area and len(approx)==4:
                          biggest = approx
                          max_area = area
      cv2.drawContours(img, biggest, -1, (0,0,255), 40)
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(img,'Press Esc If the 4 dots are at the corners of puzzle',(height/2-100,width/2-200), font, 0.7,(0,0,255),2,cv2.LINE_AA)
      cv2.imshow('img',img)
      k = cv2.waitKey(10) & 0xFF
      if k == 27:
            break


biggest=rectify(biggest)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)
retval = cv2.getPerspectiveTransform(biggest,h)
warp = cv2.warpPerspective(thresh,retval,(450,450))
warp2 = cv2.warpPerspective(img,retval,(450,450))



cv2.imwrite('box.jpg',warp)


puzzle = warp2.copy()
puzzlecopy = puzzle.copy()
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(warp2,'Press Any Key To Solve',(50,250), font, 1,(0,0,255),2,cv2.LINE_AA)
cv2.imshow('Detected Box',warp2)
cv2.waitKey(0)
cv2.destroyWindow('Detected Box')

# mask=cv2.imread('mask.jpg')
# mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
# ret, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
# i was trying something else!
SudokuBoard = [[0 for x in range(9)] for y in range(9)]
printingBoard = [[0 for x in range(9)] for y in range (9)]
InitialBoard = [[0 for x in range(9)]for y in range(9)]
              
for x in range (0,9):
    for y in range (0,9):
          morph=warp[(50*x):((50*x)+50),(50*y):((50*y)+50)]
          morph=morph[5:45,5:45]    
          morph=255-morph
          cv2.imwrite('sudokuDigits/cell'+str(x)+str(y)+'.jpg',morph)

print("Reconizing Numbers")
keys = [i for i in range(48,58)]
for x in range(0,9):
    for y in range(0,9):
        im = Image.open('C:/Users/Sul4/Desktop/Projects/sudoku/sudokuDigits/cell' + str(x) + str(y) + '.jpg')
        text = pytesseract.image_to_string(im,config='-psm 6 digits')
        if text=='\n' or text=='' or text==' ' or  ord(text[:1]) not in keys:
            SudokuBoard[x][y] = 0
        else:
            SudokuBoard[x][y] = int(text[:1])


printBoard(SudokuBoard)
InitialBoard = SudokuBoard
sudokuSolver(SudokuBoard,printingBoard)
cv2.waitKey(0)
cv2.destroyAllWindows()

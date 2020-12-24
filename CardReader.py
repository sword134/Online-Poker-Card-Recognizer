import pyautogui
import win32gui
import numpy as np
import cv2
import win32process
from pywinauto import Desktop
import imutils
import argparse

def create_template(suit, number):
    im_prefix = 'D:\PokerAlgo\images\cards_m'
    # Adjust the left and right positions of the numbers. pixel
    shift = {
        's': [-1, -1, 0, -1, -1, 0, -1, -1, -1, 0, -1, 0, 0],
        'h': [-1, -1, 0, -1, -1, 0, -1, -1, -1, 0, -1, 0, 0],
        'd': [-1, -1, 0, -1, -1, 0, -1, -1, -1, 0, -1, 0, 0],
        'c': [-1, -1, 0, -1, -1, 0, -1, -1, -1, 0, -1, 0, 0],
    }
    #space = {'s': 4, 'h': 6, 'd': 4, 'c': 2}
    space = {'s': 8, 'h': 10, 'd': 6, 'c': 8}
    color = {'s': 'b', 'h': 'r', 'd': 'r', 'c': 'b'}
    n = cv2.imread('{}/{}{}.png'.format(im_prefix, number, color[suit]))
    s = cv2.imread('{}/{}.png'.format(im_prefix, suit))[1:-1, :]

    s_width = s.shape[1]
    s_height = s.shape[0]
    # Add 2px margin for position adjustment to the number part
    '''
    template_width = max(s.shape[1], n.shape[1] + 2)

    n_back = np.full((n.shape[0], template_width, 3), 255)
    s_back = np.full((s_height, template_width, 3), 255)
    spacer = np.full((space.get(suit), template_width, 3), 255)

    n_width = n.shape[1]
    n_left_margin = int((n_back.shape[1] - n_width)/2) + shift[suit][number - 1]
    s_left_margin = int((s_back.shape[1] - s_width)/2)

    n_back[:, n_left_margin:n_left_margin + n.shape[1]] = n
    s_back[:, s_left_margin:s_left_margin + s.shape[1]] = s

    n_and_s = np.vstack((n_back, spacer, s_back)).astype('u1')
    '''
    template_width = max(s.shape[1], n.shape[1]+2)

    n_back = np.full((n.shape[0], template_width, 3), 255)
    s_back = np.full((s_height, template_width, 3), 255)
    spacer = np.full((space.get(suit), template_width, 3), 255)

    n_width = n.shape[1]
    n_left_margin = int((n_back.shape[1] - n_width)/2) + shift[suit][number - 1]
    s_left_margin = int((s_back.shape[1] - s_width)/2)

    n_back[:, n_left_margin:n_left_margin + n.shape[1]] = n
    s_back[:, s_left_margin:s_left_margin + s.shape[1]] = s

    n_and_s = np.vstack((n_back, s_back)).astype('u1')

    template = cv2.cvtColor(n_and_s, cv2.COLOR_BGR2GRAY)
    #return template

    width = 110
    height = 200
    scaled  =  cv2.resize(template,(width,height),cv2.INTER_CUBIC)
    # downscale
    #scaled = cv2.resize(template, (int(n_and_s.shape[1]*1.5), int(n_and_s.shape[0]*1.5)), interpolation=cv2.INTER_AREA)
    #cv2.imshow('image',scaled)
    #cv2.waitKey(0)
    return scaled

def screenshot(window_title=None):
    if window_title:
        hwnd = win32gui.FindWindow(None, window_title)
        if hwnd:
            win32gui.SetForegroundWindow(hwnd)
            x, y, x1, y1 = win32gui.GetClientRect(hwnd)
            x, y = win32gui.ClientToScreen(hwnd, (x, y))
            x1, y1 = win32gui.ClientToScreen(hwnd, (x1 - x, y1 - y))
            im = pyautogui.screenshot(region=(x, y, x1, y1))
            return im
        else:
            print('Window not found!')
    else:
        im = pyautogui.screenshot()
        return im

#def winEnumHandler( hwnd, ctx ):
#    thelister = []
#    if win32gui.IsWindowVisible( hwnd ):
#        print (win32gui.GetWindowText( hwnd ))



if __name__ == '__main__':   
    windows = Desktop(backend="uia").windows()
    windowsinlist = [w.window_text() for w in windows]
    print(windowsinlist)

    Word = "Pleasureville"
    shittostring = [i for i in windowsinlist if Word in i] 

    shittostringconv = str(shittostring[0])
    print(shittostringconv)

    active_window_info = screenshot(shittostringconv)
    yeet = screenshot(shittostringconv)
    
    width, height = active_window_info.size   # Get dimensions
    new_width = 600
    new_height = 150
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    active_window_info = active_window_info.crop((left, top, right, bottom))    

    #out_image = cv2.imread('D:\PokerAlgo\images_for_test\poker.png') #Use this if you want to test it on the client but you dont have it open
    
    out_image = np.array(active_window_info)
    dim = (1400, 500)
    out_image = cv2.resize(out_image, dim, interpolation = cv2.INTER_AREA)
    
    window_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2GRAY)

    #cv2.imshow('image',np.array(active_window_info))
    #cv2.waitKey(0)

    detected_cards = []

    for s in 'shdc':
        for i in range(1,14):
            template = create_template(s, i)
            result = cv2.matchTemplate(window_image, template, cv2.TM_CCOEFF_NORMED)
            
            # cv2.imwrite('template_{}_{}.png'.format(i, s), template)

            # Get the position of the detection area from the detection result
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            # print('s={}, i={}'.format(s, i))
            # print('min_val={:.3f}, max_val={:.3f}, min_loc={}, max_loc={}'.format(min_val, max_val, min_loc, max_loc))

            if max_val < 0.10:
                continue

            detected_cards.append([s, i, max_val])

            # top_left = max_loc
            # w, h = template.shape[::-1]
            # bottom_right = (top_left[0] + w, top_left[1] + h)
            #
            # cv2.rectangle(out_image, top_left, bottom_right, (255, 0, 0), 2)

    suit_dict = {'s': 'S', 'h': 'H', 'd': 'D', 'c': 'C'}
    num_dict = {1: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q', 13: 'K'}

    num_cards = len(detected_cards)
    print('Number of cards: {}'.format(num_cards))

    if num_cards == 0:
        print('Stage: --')
    elif 1 <= num_cards <= 2:
        print('Stage: Pre-Flop')
    elif 3 <= num_cards <= 5:
        print('Stage: Flop')
    elif num_cards == 6:
        print('Stage: Turn')
    elif num_cards == 7:
        print('Stage: River')

    message = '\n'.join(map(lambda x: '{}-{},\tsimilarity={:.3f}'. format(num_dict[x[1]], suit_dict.get(x[0]), x[2]), detected_cards))
    print(message)
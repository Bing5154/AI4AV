import pygame
import time
 
pygame.init()

acceleration = 0
A = joystick.get_button(0) #get the current button state, 0-off, 1-on
B = joystick.get_button(1) 

while (A == 0): 
    pygame.event.get()
    joystick_count = pygame.joystick.get_count() #get the number of joytsticks
    joystick = pygame.joystick.Joystick(0) #get the first joystick object
    joystick.init()  #initialize

    A = joystick.get_button(0) #get the current button state, 0-off, 1-on
    B = joystick.get_button(1) 
    posa = joystick.get_axis(0)
    posb = joystick.get_axis(1)

    #print(A,B,posa,posb)

    if(posa > 0 and posb > 0):
        print("You are moving backward right", end="")
        acceleration += posa
        print ("at ", acceleration, " mph")
    elif(posa > 0 and posb < 0):
        print("You are moving forward right",end="")
        acceleration += posa
        print ("at ", acceleration, " mph")
    elif(posa < 0 and posb < 0):
        print("You are moving forward left",end="")
        acceleration += posa
        print ("at ", acceleration, " mph")
    elif(posa < 0 and posb > 0):
        print("You are moving backward left",end="")
        acceleration += posa
        print ("at ", acceleration, " mph")
    elif(posa == 0 and posb ==  0.999969482421875):
        print("You are moving directly backward", end = "")
        acceleration += posb
        print ("at ", acceleration, " mph")
    elif(posa == 0 and posb ==-1):
        print("You are moving directly forward")
        acceleration += posb
        print ("at ", acceleration, " mph")
    elif(posa == 0.999969482421875 and posb == 0):
        print("You are moving directly to the right", end ="")
	acceleration += posa
        print ("at ", acceleration, " mph")
    elif(posa == -1 and posb == 0):
        print("You are moving directly to the left", end ="")
        acceleration += posa
        print ("at ", acceleration, " mph")
    elif(posa == 0 and posb == 0):
        print("You are not moving at all", end = "")
        acceleration += posa
        print ("at ", acceleration, " mph")
 






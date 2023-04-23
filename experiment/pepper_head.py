# https://answers.ros.org/question/323082/stabilizing-peppers-head/
import time
import almath
import argparse
from naoqi import ALProxy
from pprint import pprint

tts = None
motion = None
posture = None
autonomouslife = None
fractionMaxSpeed = 0.1


def init(robotIP, PORT):
    global tts, motion, posture, autonomouslife
    tts = ALProxy("ALTextToSpeech", robotIP, PORT)
    autonomouslife = ALProxy("ALAutonomousLife", robotIP, PORT)
    motion = ALProxy("ALMotion", robotIP, PORT)
    posture = ALProxy("ALRobotPosture", robotIP, PORT)


def say(something_to_say):
    tts.say(str(something_to_say))

def stopMove():
    say('Stop here')
    posture.stopMove()
    time.sleep(4.0)
    say('I am ready')

def standup():
    say('Standing up')
    posture.goToPosture("Stand", 1)
    time.sleep(4.0)
    say('I am ready')


def set_head_stiffness():
    value = 1.0
    say('Setting heading stiffness to ' + str(value))
    motion.setStiffnesses("Head", value)
    time.sleep(4.0)


def head_yaw(angle):
    names = "HeadYaw"
    angles = angle * almath.TO_RAD
    say('setting head yaw to ' + str(angle) + 'degrees')
    # motion.setAngles(names, angles, fractionMaxSpeed)
    motion.angleInterpolation(names, angles, 3.0, True)
    time.sleep(4.0)
    say('done head yaw')


def head_pitch(angle):
    names = "HeadPitch"
    angles = angle * almath.TO_RAD
    say('setting head pitch to ' + str(angle) + 'degrees')
    # motion.setAngles(names, angles, fractionMaxSpeed)
    motion.angleInterpolation(names, angles, 3.0, True)
    #time.sleep(4.0)
    say('done head pitch')


def check_auto_life():
    st = autonomouslife.getAutonomousAbilitiesStatus()
    pprint(st)
    print('\n\n')

def main(pitch, yam, robotIP, PORT=9559):
    init(robotIP, PORT)
    #autonomouslife.setState('interactive')
    
    #al_state = autonomouslife.getState()
    #if al_state != 'disabled':
    #    say('auto life is ' + al_state)
    #    
    standup()
    #tts.say('auto life is ' + autonomouslife.getState())
    #stopMove()
    #check_auto_life()
    #tts.say('auto life is ' + autonomouslife.getState())
    
    set_head_stiffness()
    #check_auto_life()
    #tts.say('auto life is ' + autonomouslife.getState())
    
    head_pitch(pitch)
    #check_auto_life()
    #tts.say('auto life is ' + autonomouslife.getState())
    
    head_yaw(yam)
    #check_auto_life()
    #tts.say('auto life is ' + autonomouslife.getState())

    autonomouslife.setState('solitary')
    w = 2
    say('Waiting for ' + str(w) + ' seconds')
    check_auto_life()
    time.sleep(w)
    say('Bye bye ')
    check_auto_life()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.20.10.10", help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559, help="Robot port number")
    parser.add_argument("--pitch", type=float, default=0, help="Head pitch in degrees")
    parser.add_argument("--yam", type=float, default=0, help="Head yam in degrees")
    
    args = parser.parse_args()
    main(args.pitch, args.yam, args.ip, args.port)

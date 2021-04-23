from SocialNetwork import SocialNetwork
import matplotlib.pyplot as plt
import numpy as np

def main():

    props = {}
    props['n'] = 50
    props['topology'] = 'small world'
    props['saturation'] = .1
    props['rewire'] = .1
    props['seed'] = None
    props['weight'] = 1.
    props['dimensions'] = 4
    props['unfriend'] = .9
    props['unfriend_threshold'] = .5
    props['friend'] = .1
    props['update'] = 1.
    props['visibility'] = 'visible'
    props['symmetric'] = True
    props['type_dist'] = { 'R':.5, 'E' : .5, 'DA':.0,
                           'RWC':.0, 'SC':.0, 'SR':.0 }
    props['resistance_param'] = 0.
    
    G = SocialNetwork( props )

    ## This will bring up the debugger mode that lets you carry out a
    ## simulation one step at a time, and monitor or set any values as you go.
    ## Type 'help' for a list of possible commands.
    G.debug()
    
##    for step in range( STEPS ):
##        G.step()

if __name__ == '__main__':
    main()

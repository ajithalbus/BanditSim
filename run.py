import pickle as pkl
from algorithms import runEpsilonGreedy,runSoftMax,runUCB,runMedianElimination
import argparse
import sys
import time
import os

def checkArgs(args=None):
    parser = argparse.ArgumentParser(description='Bandit alogorithms')
    parser.add_argument('--algo',help = 'algorithm [eg,sm,ucb,me]', default = 'eg')
    parser.add_argument('--para',help = r'parameters [$\epsilon$,$\tau$,c] if multiple,seperate with comma', default = '0.1')
    parser.add_argument('--steps',help = 'No. of steps', default = '1000')
    parser.add_argument('--exps', help = 'No. of experiments', default = '2000')
    parser.add_argument('--k',help = 'No. of arms', default = '10')
    parser.add_argument('--pacEPSILON',help='pac bounds only for ME, epsilon values [seperate by comma]',default='0.5')
    parser.add_argument('--pacDELTA',help='pac bounds only for ME, delta value [only one value]',default='0.1')
    
    args = parser.parse_args(args)
    return args


def save_weights(list_of_weights):
    f=open('./results.pkl', 'w')
    pkl.dump(list_of_weights, f)

def load_weights():
    f=open('./results.pkl')
    list_of_weights = pkl.load(f)
    return list_of_weights

if __name__ == "__main__":
    start_time = time.time()
    
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    args = checkArgs(sys.argv[1:])
    if args.algo=='eg':
        runEpsilonGreedy( k=int(args.k),steps=int(args.steps),eps=map(float,args.para.split(',')),exps=int(args.exps))
    elif args.algo=='sm':
        runSoftMax( k=int(args.k),steps=int(args.steps),tau=map(float,args.para.split(',')),exps=int(args.exps))
    elif args.algo=='ucb':
        runUCB( k=int(args.k),steps=int(args.steps),C=map(float,args.para.split(',')),exps=int(args.exps))
    elif args.algo=='me':
        runMedianElimination(k=int(args.k),eps=map(float,args.pacEPSILON.split(',')),delta=float(args.pacDELTA),exps=int(args.exps))
    print 'plots saved in ./plots'
    print "-- Running Time : %s s --" % (time.time() - start_time)
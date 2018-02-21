# BanditSim
Multi-Armed Bandits simulator and related Reinforcement Learning algorithms 
----------------------
Command Line Arguments
----------------------

Example:  $python run.py --algo eg --para 0.1,0.01 --exps 2000 --steps 1000 --k 10

    --algo eg | sm | ucb | me

    
        eg : Epsilon-Greedy
        sm : Softmax
        ucb: UCB1
        me : Median Elimination Algorithm
        Default - eg
    

    --para parameters for the algorithm [seperated by commas if multiple]

        eg : epsilon value(s)
        sm : tau value(s)
        ucb: c value(s)
        not required for MEA
        Default - 0.1
     
    --steps #steps

        Number of steps to run [not needed for MEA]
        Default - 1000

    --exps #experiments

        Number of experiments to run
        Default - 2000

    --k #arms

        Number of arms for the testbed
        Default - 10

    --pacEPSILON epsilon [only for ME,seperated by commas if multiple]

        Epsilon value(s) for pac guarantee
        Default - 0.5


    --pacDELTA delta [only for ME,only one value]

        Delta value for pac guarantee
        Default - 0.1

------------
Requirements
------------

pickle      : for storing the results if needed

argparse    : for parsing command line arguments

numpy       : for numerical computations

progressbar : for progress bar

matplotlib  : for ploting the results


--------
Examples
--------

1) Epsilon-greedy

    $python run.py --algo eg --para 0.1,0.01,0.0 --exps 2000 --steps 1000 --k 10

2) Softmax

    $python run.py --algo sm --para 0.5,0.1,0.001 --exps 2000 --steps 1000 --k 10

3) UCB1

    $python run.py --algo ucb --para 2,1,0.1 --exps 2000 --steps 1000 --k 10

4) MEA

    $python run.py --algo me --pacEPSILON 0.5 --pacDELTA 0.2 --exps 2000 --k 10


---------
Reference
---------


    [Action Elimination and Stopping Conditions for the
Multi-Armed Bandit and Reinforcement Learning Problems](http://jmlr.csail.mit.edu/papers/volume7/evendar06a/evendar06a.pdf)


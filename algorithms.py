import numpy as np
import matplotlib.pyplot as plt
import sys
from testbed import Testbed
# coding: utf-8

# In[1]:


        


# In[3]:


#Q1
class EpsilonGreedy:
    
    def __init__(self,epsilon,k=10,steps=1000):
        self.epsilon=epsilon
        self.k=k
        self.steps=steps
        self.testbed=Testbed(k)
        self.Q=np.zeros(k) #estimates
        self.N=np.zeros(k,dtype=int) #arm counter
        self.Policy=np.zeros(k) #policy PI
        
    def calculatePolicy(self):
        aStar=np.random.choice(np.flatnonzero(self.Q == self.Q.max())) #argmax tie-breaker
        for i in range(self.k):
            self.Policy[i]=self.epsilon/self.k
        self.Policy[aStar]=1-self.epsilon+self.epsilon/self.k
    
    def updateQ(self,arm,reward):
        self.Q[arm]=self.Q[arm]+(1.0/self.N[arm])*(reward-self.Q[arm])
    
    def selectArm(self):
        return np.random.choice(self.k,p=self.Policy)
    def run(self):
        rewards=[] #list of recieved rewards
        optimalAction=[]
        optimalArm=self.testbed.optimalArm()
        for i in range(self.steps):
            self.calculatePolicy()
            arm=self.selectArm() 
            self.N[arm]+=1
            reward=self.testbed.pull(arm)
            rewards.append(reward)
            optimalAction.append(1*(arm==optimalArm))
            self.updateQ(arm,reward)
            
        #optimalAction=np.cumsum(optimalAction,dtype=np.float)/range(1,self.steps+1)
        return np.array(rewards),np.array(optimalAction)


# In[4]:

def runEpsilonGreedy(k=10,steps=1000,eps=[0.1],exps=2000):
    rev_avgs=[] #stores average rewards 
    oa_avgs=[] #stores optimal action % avgs
    for epsilon in eps:
        rewards=[]
        optimalActions=[]
        for i in range(exps):
            print i
            epsilonGreedy=EpsilonGreedy(epsilon=epsilon,steps=steps,k=k)
            reward,optimalAction=epsilonGreedy.run()
            rewards.append(reward)
            optimalActions.append(optimalAction)
        rewards=np.array(rewards)
        optimalActions=np.array(optimalActions)
        rev_avgs.append(np.mean(rewards,axis=0))
        oa_avgs.append(np.mean(optimalActions,axis=0))
        plt.figure(1)
        plt.plot(range(steps),rev_avgs[-1],'-',label=r'$\epsilon=$'+str(epsilon))
        plt.figure(2)
        plt.plot(range(steps),oa_avgs[-1]*100,'-',label=r'$\epsilon=$'+str(epsilon))
    plt.title(r'$\epsilon$-greedy Algorithm')    
    plt.xlabel('steps')
    plt.ylabel('% Optimal Action')
    plt.legend()
    plt.savefig('./plots/1b.eps',format='eps')

    plt.figure(1)
    plt.title(r'$\epsilon$-greedy Algorithm')
    plt.xlabel('steps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.savefig('./plots/1a.eps',format='eps')
    plt.show()
    return rev_avgs,oa_avgs

class SoftMax:
    
    def __init__(self,k=10,steps=1000,temp=0.1):
        self.temp=temp
        self.k=k
        self.steps=steps
        self.testbed=Testbed(k)
        self.Q=np.zeros(k) #estimates
        self.N=np.zeros(k,dtype=int) #arm counter
        self.Policy=np.zeros(k) #policy PI
        
    def calculatePolicy(self):
        self.Policy=self.Q/self.temp
        self.Policy=np.exp(self.Policy)
        self.Policy=self.Policy/np.sum(self.Policy)
    def updateQ(self,arm,reward):
        self.Q[arm]=self.Q[arm]+(1.0/self.N[arm])*(reward-self.Q[arm])
    
    def selectArm(self):
        return np.random.choice(self.k,p=self.Policy)
    def run(self):
        rewards=[] #list of recieved rewards
        optimalAction=[]
        optimalArm=self.testbed.optimalArm()
        for i in range(self.steps):
            self.calculatePolicy()
            arm=self.selectArm() 
            self.N[arm]+=1
            reward=self.testbed.pull(arm)
            rewards.append(reward)
            optimalAction.append(1*(arm==optimalArm))
            self.updateQ(arm,reward)
            
        #optimalAction=np.cumsum(optimalAction,dtype=np.float)/range(1,self.steps+1)
        return np.array(rewards),np.array(optimalAction)


# In[ ]:

def runSoftMax(k=10,steps=1000,exps=2000,tau=[0.1]):
    rev_avgs=[]
    oa_avgs=[]
    for temp in tau:
        rewards=[]
        optimalActions=[]
        for i in range(exps):
            softmax=SoftMax(temp=temp,steps=steps,k=k)
            reward,optimalAction=softmax.run()
            rewards.append(reward)
            optimalActions.append(optimalAction)
            
        rewards=np.array(rewards)
        optimalActions=np.array(optimalActions)
        rev_avgs.append(np.mean(rewards,axis=0))
        oa_avgs.append(np.mean(optimalActions,axis=0))
        plt.figure(1)
        plt.plot(range(steps),rev_avgs[-1],'-',label=r'$\tau=$'+str(temp))
        plt.figure(2)
        plt.plot(range(steps),oa_avgs[-1]*100,'-',label=r'$\tau=$'+str(temp))

    plt.title('Softmax Algorithm')    
    plt.xlabel('steps')
    plt.ylabel('% Optimal Action')
    plt.legend()
    plt.savefig('./plots/2b.eps',format='eps')

    plt.figure(1)
    plt.title('Softmax Algorithm')    
    plt.xlabel('steps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.savefig('./plots/2a.eps',format='eps')
    plt.show()
    return rev_avgs,oa_avgs


class UCB1:
    
    def __init__(self,k=10,steps=1000,c=1):
        self.c=c
        self.k=k
        self.steps=steps
        self.testbed=Testbed(k)
        self.Q=np.zeros(k) #estimates
        self.N=np.zeros(k,dtype=int) #arm counter
        self.Policy=np.zeros(k) #policy PI
        
    def calculatePolicy(self,t):
        self.Policy=self.Q+self.c*np.sqrt(np.log(t)/self.N)
        
    def updateQ(self,arm,reward):
        self.Q[arm]=self.Q[arm]+(1.0/self.N[arm])*(reward-self.Q[arm])
    
    def selectArm(self):
        return np.argmax(self.Policy)
    def run(self):
        rewards=[] #list of recieved rewards
        optimalAction=[]
        optimalArm=self.testbed.optimalArm()
        for i in range(self.steps):
            self.calculatePolicy(i)
            arm=self.selectArm() 
            self.N[arm]+=1
            reward=self.testbed.pull(arm)
            rewards.append(reward)
            optimalAction.append(1*(arm==optimalArm))
            self.updateQ(arm,reward)
            
        #optimalAction=np.cumsum(optimalAction,dtype=np.float)/range(1,self.steps+1)
        return np.array(rewards),np.array(optimalAction)


# In[ ]:

def runUCB(k=10,steps=1000,C=[1],exps=2000):
    rev_avgs=[]
    oa_avgs=[]
    for c in C:
        rewards=[]
        optimalActions=[]
        for i in range(exps):
            ucb=UCB1(c=c,steps=steps,k=k)
            reward,optimalAction=ucb.run()
            rewards.append(reward)
            optimalActions.append(optimalAction)
            
        rewards=np.array(rewards)
        optimalActions=np.array(optimalActions)
        rev_avgs.append(np.mean(rewards,axis=0))
        oa_avgs.append(np.mean(optimalActions,axis=0))
        plt.figure(1)
        plt.plot(range(steps),rev_avgs[-1],'-',label='c='+str(c))
        plt.figure(2)
        plt.plot(range(steps),oa_avgs[-1]*100,'-',label='c='+str(c))
    
    plt.title('UCB1 Algorithm')    
    plt.xlabel('steps')
    plt.ylabel('% Optimal Action')
    plt.legend()
    plt.savefig('./plots/3b.eps',format='eps')

    plt.figure(1)
    plt.title('UCB1 Algorithm')    
    plt.xlabel('steps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.savefig('./plots/3a.eps',format='eps')
    plt.show()
    return rev_avgs,oa_avgs


class MedianElimination:
    
    def __init__(self,k=10,epsilon=0.5,delta=0.1):
        self.epsilon=epsilon
        self.k=k
        
        self.testbed=Testbed(k)
        self.Q=np.zeros(k) #estimates
        self.N=np.zeros(k,dtype=int) #arm counter
        self.delta=delta
        
    def updateQ(self,arm,reward):
        self.Q[arm]=self.Q[arm]+(1.0/self.N[arm])*(reward-self.Q[arm])
    
    def selectBadArms(self,median):
        return np.flatnonzero(self.Q<median)
    def removeArms(self,armsList):
        self.Q=np.delete(self.Q,armsList)
        self.N=np.delete(self.N,armsList)
        self.testbed.dropArms(armsList)
        self.k-=len(armsList)
    
    def run(self):
        rewards=[] #list of recieved rewards
        optimalAction=[]
        optimalArm=self.testbed.optimalArm()
        S=self.testbed # for sync with the algo
        epsilon=self.epsilon/4
        delta=self.delta/2
        #p=Pool(4)
        while S.k>1: #S.k is the number of arms 
            #print 'no of arms left:',S.k
            l=int((1/((epsilon/2)**2))*np.log(3/delta)) #magic number
            #print 'l :',l
            for arm in range(S.k):
                for i in range(l):
                    reward=S.pull(arm)
                    self.N[arm]+=1
                    rewards.append(reward)
                    optimalAction.append(1*(arm==optimalArm))
                    self.updateQ(arm,reward)
            #print self.Q
            #print self.N
            median=np.median(self.Q)
            #print 'median:',median
            badArms=self.selectBadArms(median)
            self.removeArms(badArms)
            epsilon*=0.75
            delta*=0.5
        #print 'Choosen Best Arm q : ',S.arms[0].q #) because there is only one arm left
        return np.array(rewards),np.array(optimalAction)


# In[15]:

def runMedianElimination(k=10,eps=[0.5],delta=0.1,exps=2000):
    rev_avgs=[]
    oa_avgs=[]
    for epsilon in eps:
        rewards=[]
        optimalActions=[]
        for i in range(exps):
            medianElimination=MedianElimination(k=k,epsilon=epsilon,delta=delta)
            reward,optimalAction=medianElimination.run()
            rewards.append(reward)
            optimalActions.append(optimalAction)
            #print reward
        rewards=np.array(rewards)
        optimalActions=np.array(optimalActions)
        rev_avgs.append(np.mean(rewards,axis=0))
        oa_avgs.append(np.mean(optimalActions,axis=0))
        plt.figure(1)
        plt.plot(range(len(rev_avgs[-1])),rev_avgs[-1],'-',label='epsilon='+str(epsilon))
        plt.figure(2)
        plt.plot(range(len(oa_avgs[-1])),oa_avgs[-1]*100,'-',label='epsilon='+str(epsilon))
    plt.title(r"Median elimination algorithm $\delta=$"+str(delta))    
    plt.xlabel('steps')
    plt.ylabel('% Optimal Action')
    plt.legend()
    plt.savefig('./plots/4b.eps',format='eps')

    plt.figure(1)
    plt.title(r"Median elimination algorithm $\delta=$"+str(delta))    
    plt.xlabel('steps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.savefig('./plots/4a.eps',format='eps')
    plt.show()




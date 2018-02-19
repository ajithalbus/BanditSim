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


rev_avgs=[] #stores average rewards 
oa_avgs=[] #stores optimal action % avgs
for epsilon in [0.1,0.01,0.0]:
    rewards=[]
    optimalActions=[]
    for i in range(2000):
        epsilonGreedy=EpsilonGreedy(epsilon=epsilon,steps=1000,k=1000)
        reward,optimalAction=epsilonGreedy.run()
        rewards.append(reward)
        optimalActions.append(optimalAction)
    rewards=np.array(rewards)
    optimalActions=np.array(optimalActions)
    rev_avgs.append(np.mean(rewards,axis=0))
    oa_avgs.append(np.mean(optimalActions,axis=0))
    plt.figure(1)
    plt.plot(range(1000),rev_avgs[-1],'-',label=r'$\epsilon=$'+str(epsilon))
    plt.figure(2)
    plt.plot(range(1000),oa_avgs[-1]*100,'-',label=r'$\epsilon=$'+str(epsilon))
    
plt.xlabel('steps')
plt.ylabel('% Optimal Action')
plt.legend()
plt.savefig('./plots/1b.eps',format='eps')

plt.figure(1)
plt.xlabel('steps')
plt.ylabel('Average Reward')
plt.legend()
plt.savefig('./plots/1a.eps',format='eps')
plt.show()



import numpy as np
#from pathos.multiprocessing import ProcessingPool as Pool
#get_ipython().magic(u'pylab inline')


# In[2]:


#TestBed
class Arm:
    def __init__(self,q):
        self.q=q
    def pull(self):
        return np.random.normal(loc=self.q)
        
class Testbed:
    def __init__(self,k=10):
        self.k=k
        self.q=np.random.normal(size=k)
        self.arms=[]
        for i in range(k):
            self.arms.append(Arm(self.q[i]))
        #print 'Best Arm q:',self.arms[self.optimalArm()].q
    def pull(self,arm):
        return self.arms[arm].pull()
    def optimalArm(self):
        return np.argmax(self.q)
    def dropArms(self,armsList): #only for median elimiation
        self.arms=np.delete(self.arms,armsList)
        self.k-=len(armsList) 
        

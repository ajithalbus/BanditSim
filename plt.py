import pickle as pkl
import matplotlib.pyplot as plt
def save_weights(list_of_weights):
    f=open('./results.pkl', 'w')
    pkl.dump(list_of_weights, f)

def load_weights():
    f=open('./results.pkl')
    list_of_weights = pkl.load(f)
    return list_of_weights
results=load_weights()
labs=[0.1,0.1,1]
strs=[r'$\epsilon$-greedy with $\epsilon=$',r'Softmax with $\tau=$','UCB1 with c=']
for i in range(len(results)/2):
    plt.figure(1)
    plt.plot(range(1000),results[2*i][0],'-',label=strs[i]+str(labs[i]))
    plt.figure(2)
    plt.plot(range(1000),results[2*i+1][0]*100,'-',label=strs[i]+str(labs[i]))

plt.xlabel('steps')
plt.ylabel('% Optimal Action')
plt.legend()
plt.savefig('./plots/compb.eps',format='eps')

plt.figure(1)
plt.xlabel('steps')
plt.ylabel('Average Reward')
plt.legend()
plt.savefig('./plots/compa.eps',format='eps')
plt.show()
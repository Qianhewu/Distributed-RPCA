import pickle
import numpy as np
import matplotlib.pyplot as plt


a_file = open("differentK.pkl", "rb")
dict = pickle.load(a_file)
a_file.close()
step = np.arange(0,51)

plt.figure(figsize=(8,5),dpi=300)
plt.title("Comparison of different RPCA algorithms")
plt.xlabel("# Iterations")
plt.ylabel("Relative err")
plt.yscale('log')
size=2
plt.plot(step,dict["DCF_500_1"]["stat"],markersize=size,label='K=1')
plt.plot(step,dict["DCF_500_3"]["stat"],markersize=size,label='K=3')
plt.plot(step,dict["DCF_500_5"]["stat"],markersize=size,label='K=5')
plt.plot(step,dict["DCF_500_10"]["stat"],markersize=size,label='K=10')
plt.legend(loc=1,fontsize=11)
plt.savefig("figures/different_K",bbox_inches='tight',dpi=300)


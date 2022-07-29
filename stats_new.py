import numpy as np
import matplotlib.pyplot as plt

# data = {"stat": x, "mu0": 5}
# dict = {"APGM_3000": data}
import pickle
# a_file = open("data.pkl", "wb")
# pickle.dump(dict, a_file)
# a_file.close()
colors = ["#56B4E9", "#009E73",  "#D55E00", "#CC79A7", "#F0E442"]
styles = ["-", "--", "-.", ":"]


a_file = open("data.pkl", "rb")
dict = pickle.load(a_file)
a_file.close()

print(dict.keys())
step = np.arange(0,300)

plt.figure(figsize=(8,5),dpi=300)
plt.title("Comparison of different RPCA algorithms")
plt.xlabel("# Iterations")
plt.xlim(0,150)
plt.ylabel("Relative err")
plt.yscale('log')
size=2
plt.plot(step[:60],dict["DCF_500"]["stat"][:60],markersize=size,label='DCF-PCA-500',color=colors[0],ls=styles[0])
plt.plot(step[:65],dict["DCF_1000"]["stat"][:65],markersize=size,label='DCF-PCA-1000',color=colors[0],ls=styles[1])
plt.plot(step[:len(dict["DCF_temp_3000"]["stat"][:50])],dict["DCF_temp_3000"]["stat"][:50],markersize=size,label='DCF-PCA-3000',color=colors[0],ls=styles[2])

# plt.plot(step,K10_M500_RS005_rank,label='DCF-PCA unknown rank')
# plt.plot(step,ALM_M500,marker='o',markersize=size,label='ALM')
plt.plot(step[:len(dict["APGM_500"]["stat"])],dict["APGM_500"]["stat"],markersize=size,label='APGM-500',color=colors[1],ls=styles[0])
plt.plot(step[:len(dict["APGM_1000"]["stat"])],dict["APGM_1000"]["stat"],markersize=size,label='APGM-1000',color=colors[1],ls=styles[1])
plt.plot(step[:len(dict["APGM_3000"]["stat"])],dict["APGM_3000"]["stat"],markersize=size,label='APGM-3000',color=colors[1],ls=styles[2])

plt.plot(step[:len(dict["ALM_500"]["stat"])-3],dict["ALM_500"]["stat"][:-3],markersize=size,label='ALM-500',color=colors[2],ls=styles[0])
plt.plot(step[:len(dict["ALM_1000"]["stat"])-2],dict["ALM_1000"]["stat"][:-2],markersize=size,label='ALM-1000',color=colors[2],ls=styles[1])
plt.plot(step[:30],dict["ALM_3000"]["stat"][:30],markersize=size,label='ALM-3000',color=colors[2],ls=styles[2])

plt.plot(step[:len(dict["CF_500"]["stat"][:100])],dict["CF_500"]["stat"][:100],markersize=size,label='CF-500',color=colors[3],ls=styles[0])
plt.plot(step[:len(dict["CF_1000"]["stat"][:100])],dict["CF_1000"]["stat"][:100],markersize=size,label='CF-1000',color=colors[3],ls=styles[1])
plt.plot(step[:len(dict["CF_3000"]["stat"])],dict["CF_3000"]["stat"],markersize=size,label='CF-3000',color=colors[3],ls=styles[2])



# plt.plot(step,CF_PCA_M500_LR1,marker='o',markersize=size,label='CF-PCA')
plt.legend(loc=(1.04,0),fontsize=11)
plt.savefig("figures/M500",bbox_inches='tight',dpi=300)



# data = {"stat": x, "num_clients":10, "tau":5, "lr":0.00005}
# dict["DCF_3000"] = data
# a_file = open("data.pkl", "wb")
# pickle.dump(dict, a_file)
# a_file.close()
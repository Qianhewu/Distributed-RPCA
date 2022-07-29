import matplotlib.pyplot as plt
import numpy as np

M,r,p = 500, 100, 200
path = "data/M{}_r{}_p{}".format(M,r,p)
# data_list_1 = np.array(recovered)
# data_list_2 = np.array(gt)
data_list_1 = np.load(path+"/singular_value.npy")
data_list_2 = np.load(path+"/singular_value_gt.npy")


def relative_singular_error(re,gt,rank):
       return np.max(np.abs(re-gt)) / gt[int(rank)]

err = relative_singular_error(data_list_1,data_list_2, r-1)
print("Relative singular value error:{:.4f}".format(err))

# plt.figure(figsize=(8,5),dpi=200)

# plt.axvline(x=len(gt)-0.5, color= 'b', ls='--')
# d_gt = []
# d_rc = []
# # plt.yscale('log')
# for i in range(len(gt)):
#     d_gt.append(0)
#     d_gt.append(data_list_2[i])
#     d_rc.append(data_list_1[i])
#     d_rc.append(0)
# idx = np.arange(len(d_gt))
# font = {'family': 'serif',
#         'weight': 'normal',
#         'style': 'italic',
#         'size': 16,
#         }
# plt.bar(idx, d_gt, 0.6,  alpha=0.5,  label = 'Ground Truth')
# plt.bar(idx, d_rc, 0.6,  alpha=0.5,  label = 'Recovered')
# plt.text(x=2,y=380,s=r"i $\leq$ r",size=14,fontdict=font)
# plt.text(x=len(gt) + 2,y=380,s=r"i $\geq$ r",size=14,fontdict=font)
# #     for index,data in enumerate(data_list):
# #         plt.text(x=index - 0.5 , y =data+0.5 , s="{:.1f}".format(data))

# # plt.xticks(idx,idx)

# # plt.legend(loc='upper right')
# plt.xlabel('i',fontdict=font,fontsize=10)
# plt.ylabel(r'$\sigma_i(L)$',rotation=0,fontsize=10)
# plt.grid(True)
# plt.legend(loc=1, borderaxespad=0,fontsize=12)
# # plt.title(title)
# plt.savefig("figures/singular_values.png", dpi=300,bbox_inches='tight')
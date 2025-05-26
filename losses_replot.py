import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import json
import matplotlib
import seaborn as sns
matplotlib.rcParams['pdf.fonttype'] = 42

matplotlib.rcParams['ps.fonttype'] = 42

sns.set_theme()


def stack_plots(folders):
    losses_array_list = []
    config_list = []
    for folder in folders:
        with open(os.path.join(folder, 'losses_array.pickle'), 'rb') as f:
            losses_array_list.append(pickle.load(f))

        with open(os.path.join(folder, 'config.json'), 'r') as f:
            config_list.append(json.load(f))
    
    lam_bar_list = np.array(config_list[0]['lams'])#1- np.array(config_list[0]['lams']) # 1 - \lambda

    #plt.errorbar(lam_bar, np.mean(results_array, axis = 0), yerr = np.std(results_array, axis = 0), fmt='o', ecolor='orangered', capsize=3 )
    for i in range(len(folders)):
        #plt.plot(lam_bar_list, np.mean(results_array_list[i], axis = 0), 'o--', label = f"f = {config_list[i]['f']}")
        plt.errorbar(lam_bar_list, np.mean(losses_array_list[i], axis = 0), yerr = np.std(losses_array_list[i], axis = 0) ,fmt= 'o--', label = f"m = {config_list[i]['m']}")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Test accuracy on local dataset")

    plt.ylim(60, 100)
    plt.legend()
    plt.savefig(f'experiments/mnist__/err_plot_n_20_m_32_dirichlet.png')





def plot_losses(folder):

    config_list = []
    with open(os.path.join(folder, 'losses_array.pickle'), 'rb') as f:
        losses_array = pickle.load(f)

    with open(os.path.join(folder, 'config.json'), 'r') as f:
        config_list.append(json.load(f))
    
    lam_bar_list = np.array([0,0.2, 0.4, 0.6, 0.8, 1])#1- np.array(config_list[0]['lams']) # 1 - \lambda

    #plt.errorbar(lam_bar, np.mean(results_array, axis = 0), yerr = np.std(results_array, axis = 0), fmt='o', ecolor='orangered', capsize=3 )

        #plt.plot(lam_bar_list, np.mean(results_array_list[i], axis = 0), 'o--', label = f"f = {config_list[i]['f']}")

    for lam in range(len(lam_bar_list)):

        plt.plot(losses_array[:,lam,0], label = rf"$\lambda = {lam_bar_list[lam]}$")
    # plt.errorbar(lam_bar_list, np.mean(losses_array[i], axis = 0), yerr = np.std(losses_array_list[i], axis = 0) ,fmt= 'o--', label = f"m = {config_list[i]['m']}")
    plt.xlabel("T")
    plt.ylabel("Loss")

    plt.legend()
    plt.savefig(f'{folder}/loss_plot.png')


#replot_reverse_order("experiments/n_20_m_64_f_4_T_200_dirichlet")./experiments/mnist/n_20_m_1024_f_5_T_200__runs_4_dirichlet_alpha_1
    
if __name__ == "__main__":
    folder  = "experiments/trash/mnist/dirichlet/n_20_m_32_f_6_T_305_runs_1_alpha_0.5_R" 
    plot_losses(folder)
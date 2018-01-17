import seaborn as sns
sns.reset_orig()
import matplotlib.pyplot as plt
from matplotlib import lines
from seaborn import xkcd_rgb as xkcd
import os
import numpy as np


def plot_returns(true_returns, estimated_returns, x_desired=None,
                 fig_path=os.getcwd(), show_fig=False, save_fig=False):
    """Plotting the estimated, true, and desired returns for players over the iterations.
    
    :param true_returns: (num_iterations x num_players) numpy array of true returns for each iteration.
    :param estimated_returns: (num_iterations x num_players) numpy array of estimated returns for each iteration.
    :param x_desired: (1 x num_players) numpy array of desired returns for the players.
    :params fig_path, show_fig, savefig: path to save fig, whether to show, whether to save.
    """

    x1 = estimated_returns[:, 0]
    x2 = estimated_returns[:, 1]
    
    true1 = true_returns[:, 0]
    true2 = true_returns[:, 1]

    sns.set(rc={'text.usetex' : True}, font_scale=1.0)
    sns.set_style("whitegrid", 
                 {'font.family':['serif'],
                  'font.serif':['Times New Roman']})
    
    fig = plt.figure(1,figsize=(6.5,6))
    fig.clf()
    ax = fig.add_subplot(111)

    lw = 5.5
    lw_ = 8
    fs1 = 22
    fs2 = 24
    fs3 = 26

    if x_desired is not None:
        ax.axhline([x_desired[0, 0]], color=xkcd['yellow orange'], lw=lw_)
        hline1 = lines.Line2D([], [], lw=lw_, linestyle="-", color=xkcd['yellow orange'])

    ax.plot(true1, color=xkcd['tomato red'], linestyle='-', lw=lw)
    line1 = lines.Line2D([], [], lw=lw, linestyle="-", color=xkcd['tomato red'])

    # Creating dashed line with black and tomato red for estimated return.
    ax.plot(x1, color=xkcd['tomato red'], linestyle='-', lw=lw)
    ax.plot(x1, color=xkcd['black'], linestyle='--', lw=lw)
    dotted_line1 = lines.Line2D([], [], lw=lw, linestyle="--", dashes=(10, 1), color=xkcd['tomato red'])
    dotted_line2 = lines.Line2D([], [], lw=lw, linestyle="-", dashes=(5, 4), color=xkcd['black'])

    if x_desired is not None:
        ax.axhline([x_desired[0, 1]], color=xkcd['purple'], lw=lw_)
        hline2 = lines.Line2D([], [], lw=lw_, linestyle="-", color=xkcd['purple'])

    ax.plot(true2, color=xkcd['muted blue'], linestyle='-', lw=lw)
    line2 = lines.Line2D([], [], lw=lw, linestyle="-", color=xkcd['muted blue'])

    # Creating dashed line with black and muted blue for estimated return.
    ax.plot(x2, color=xkcd['muted blue'], linestyle='-', lw=lw)
    ax.plot(x2, color=xkcd['black'], linestyle='--', lw=lw)
    dotted_line3 = lines.Line2D([], [], lw=lw, linestyle="--", dashes=(10, 1), color=xkcd['muted blue'])
    dotted_line4 = lines.Line2D([], [], lw=lw, linestyle="-", dashes=(5, 4), color=xkcd['black'])

    ax.set_title('Agent Response', fontsize=fs2)
    ax.set_xlabel('iteration', fontsize=fs2)
    ax.set_ylabel(r'$x_i$', fontsize=fs3, rotation=0, labelpad=30)

    if x_desired is not None:
        lgd = ax.legend([(hline1, hline1), (line1, line1), (dotted_line1, dotted_line2), 
                         (hline2, hline2), (line1, line2), (dotted_line3, dotted_line4)], 
                        [r'$x_1^d$', r'$x_1$', r'$\hat{x}_1$', r'$x_2^d$', r'$x_2$', r'$\hat{x}_2$'], 
                        loc='best', ncol=2, fancybox=True, fontsize=fs1)
    else:
        lgd = ax.legend([(line1, line1), (dotted_line1, dotted_line2), 
                         (line1, line2), (dotted_line3, dotted_line4)], 
                        [r'$x_1$', r'$\hat{x}_1$', r'$x_2$', r'$\hat{x}_2$'], 
                        loc='best', ncol=2, fancybox=True, fontsize=fs1)

    ax.tick_params(labelsize=fs2)
    ax.set_xlim([-20, len(estimated_returns)+20])
    fig.canvas.draw()
    
    if save_fig:
        plt.savefig(os.path.join(fig_path, 'returns.pdf'), bbox_extra_artists=(lgd,), 
                    bbox_inches='tight', dpi=300)
        
    if show_fig:
        plt.show()
    else:
        plt.close()
        
    sns.reset_orig()


def plot_estimation_error(estimation_errors, labels=[''], colors=None, 
                          fig_path=os.getcwd(), show_fig=False, save_fig=False):
    """Plotting distance between the true and estimated response for players over iterations.

    :param estimation_errors: List of numpy arrays each containing the estimation
    error for an agent behavior at each iteration given by \|x_i - \hat{x}_i\|.
    :param labels: List of labels corresponding to items in estimation_errors for plot.
    :param colors: List of colors corresponding to items in estimation_errors for plot.
    :params fig_path, show_fig, savefig: path to save fig, whether to show, whether to save.
    """ 

    sns.set(rc={'text.usetex' : True}, font_scale=1.0)
    sns.set_style("whitegrid", 
                 {'font.family':['serif'],
                  'font.serif':['Times New Roman']})
    
    fig = plt.figure(1,figsize=(6.5,6))
    fig.clf()
    ax = fig.add_subplot(111)

    lw = 5.5
    fs1 = 22
    fs2 = 24
    fs3 = 26

    for i in xrange(len(estimation_errors)):
        if colors is not None:
            ax.plot(estimation_errors[i], label=labels[i], color=colors[i], linestyle='-', lw=lw)
        else:
            ax.plot(estimation_errors[i], label=labels[i], linestyle='-', lw=lw)

    ax.set_title('Estimation Error', fontsize=fs2)
    ax.set_xlabel('iteration', fontsize=fs2)
    ax.set_ylabel(r'$\|x - \hat{x}\|_2$', fontsize=fs3, rotation='vertical')

    if len(labels) == 1 and '' in labels:
        # No legend if only plotting for one play style.
        pass
    else:
        lgd = ax.legend(loc='best', fancybox=True, fontsize=fs1)
    
    ax.tick_params(labelsize=fs2)
    ax.set_xlim([-20, len(estimation_errors[0])+20])
    fig.canvas.draw()
    
    if save_fig:
        # No legend if only plotting for one play style.
        if len(labels) == 1 and '' in labels:
            plt.savefig(os.path.join(fig_path, 'estimation_error.pdf'), 
                        bbox_inches='tight', dpi=300)
        else:
            plt.savefig(os.path.join(fig_path, 'estimation_error.pdf'), bbox_extra_artists=(lgd,), 
                bbox_inches='tight', dpi=300)

    if show_fig:
        plt.show()
    else:
        plt.close()
        
    sns.reset_orig()


def plot_parameter_error(parameter_error, fig_path=os.getcwd(), show_fig=False, 
                         save_fig=False):
    """Plotting distance between the true and estimated parameters for players over iterations.

    :param parameter_error: (num iterations x num players) numpy array containing 
    the parameter error for players at each iteration given by \|\theta_i - \hat{\theta}_i\|.
    :param labels: List of labels corresponding to items in estimation_errors for plot.
    :param colors: List of colors corresponding to items in estimation_errors for plot.
    :params fig_path, show_fig, savefig: path to save fig, whether to show, whether to save.
    """ 

    parameter_error_1 = parameter_error[:, 0]
    parameter_error_2 = parameter_error[:, 1] 

    sns.set(rc={'text.usetex' : True}, font_scale=1.0)
    sns.set_style("whitegrid", 
                 {'font.family':['serif'],
                  'font.serif':['Times New Roman']})
    
    fig = plt.figure(1,figsize=(6.5,6))
    fig.clf()
    ax = fig.add_subplot(111)

    lw = 5.5
    fs1 = 22
    fs2 = 24
    fs3 = 26

    ax.plot(parameter_error_1, label=r'$\|\theta_1 - \hat{\theta}_1\|_2$', 
            color=xkcd['tomato red'], linestyle='-', lw=lw)
    ax.plot(parameter_error_2, label=r'$\|\theta_2 - \hat{\theta}_2\|_2$', 
            color=xkcd['muted blue'], linestyle='--', lw=lw)

    ax.set_title('Parameter Error', fontsize=fs2)
    ax.set_xlabel('iteration', fontsize=fs2)
    ax.set_ylabel(r'$\|\theta_i - \hat{\theta}_i\|_2$', fontsize=fs3, rotation='vertical')

    lgd = ax.legend(loc='best', fancybox=True, fontsize=fs1)
    
    ax.tick_params(labelsize=fs2)
    ax.set_xlim([-20, len(parameter_error)+20])
    fig.canvas.draw()
    
    if save_fig:
        plt.savefig(os.path.join(fig_path, 'parameter_error.pdf'), bbox_extra_artists=(lgd,), 
                    bbox_inches='tight', dpi=300)
        
    if show_fig:
        plt.show()
    else:
        plt.close()
        
    sns.reset_orig()


def plot_incentive_error(incentive_errors, labels=[''], colors=None, 
                         fig_path=os.getcwd(), show_fig=False, save_fig=False):
    """Plotting distance to the desired response for players over iterations.

    :param incentive_errors: List of numpy arrays like each containing the incentive
    error for an agent behavior at each iteration given by \|x_i - x_i^d\|.
    :param labels: List of labels corresponding to items in incentive_errors for plot.
    :param colors: List of colors corresponding to items in incentive_errors for plot.
    :params fig_path, show_fig, savefig: path to save fig, whether to show, whether to save.
    """

    sns.set(rc={'text.usetex' : True}, font_scale=1.0)
    sns.set_style("whitegrid", 
                 {'font.family':['serif'],
                  'font.serif':['Times New Roman']})
    
    fig = plt.figure(1,figsize=(6.5,6))
    fig.clf()
    ax = fig.add_subplot(111)

    lw = 5.5
    fs1 = 22
    fs2 = 24
    fs3 = 26

    for i in xrange(len(incentive_errors)):
        if colors is not None:
            ax.plot(incentive_errors[i], label=labels[i], color=colors[i], linestyle='-', lw=lw)
        else:
            ax.plot(incentive_errors[i], label=labels[i], linestyle='-', lw=lw)

    ax.set_title('Incentive Error', fontsize=fs2)
    ax.set_xlabel('iteration', fontsize=fs2)
    ax.set_ylabel(r'$\|x - x^d\|_2$', fontsize=fs3, rotation='vertical')

    if len(labels) == 1 and '' in labels:
        # No legend if only plotting for one play style.
        pass
    else:
        lgd = ax.legend(loc='best', fancybox=True, fontsize=fs1)
    
    ax.tick_params(labelsize=fs2)
    ax.set_xlim([-20, len(incentive_errors[0])+20])
    fig.canvas.draw()

    
    if save_fig:
        # No legend if only plotting for one play style.
        if len(labels) == 1 and '' in labels:
            plt.savefig(os.path.join(fig_path, 'incentive_error.pdf'), 
                        bbox_inches='tight', dpi=300)
        else:
            plt.savefig(os.path.join(fig_path, 'incentive_error.pdf'), bbox_extra_artists=(lgd,), 
                        bbox_inches='tight', dpi=300)
        
    if show_fig:
        plt.show()
    else:
        plt.close()
        
    sns.reset_orig()


def plot_persistence_excitation(excitations, fig_path=os.getcwd(), 
                                show_fig=False, save_fig=False):
    """Plotting the persistence of excitation for players over iterations.

    :param excitations: (num_iterations x num_players) numpy array of norm 
    squared of regression vectors for each iteration.
    :params fig_path, show_fig, savefig: path to save fig, whether to show, whether to save.
    """

    excitation_1 = excitations[:, 0]
    excitation_2 = excitations[:, 1]    

    sns.set(rc={'text.usetex' : True}, font_scale=1.0)
    sns.set_style("whitegrid", 
                 {'font.family':['serif'],
                  'font.serif':['Times New Roman']})
    
    fig = plt.figure(1,figsize=(6.5,6))
    fig.clf()
    ax = fig.add_subplot(111)

    lw = 5.5
    fs1 = 22
    fs2 = 24
    fs3 = 26

    ax.plot(excitation_1, label=r'$\|\xi_1\|_2^2$', color=xkcd['tomato red'], linestyle='-', lw=lw)
    ax.plot(excitation_2, label=r'$\|\xi_2\|_2^2$', color=xkcd['muted blue'], linestyle='--', lw=lw)

    ax.set_title('Persistence of Excitation', fontsize=fs2)
    ax.set_xlabel('iteration', fontsize=fs2)
    ax.set_ylabel(r'$\|\xi_i\|_2^2$', fontsize=fs3, rotation=0, labelpad=30)

    lgd = ax.legend(loc='best', fancybox=True, fontsize=fs1)
    
    ax.tick_params(labelsize=fs2)
    ax.set_xlim([-20, len(excitations)+20])
    fig.canvas.draw()

    
    if save_fig:
        plt.savefig(os.path.join(fig_path, 'excitation.pdf'), bbox_extra_artists=(lgd,), 
                    bbox_inches='tight', dpi=300)
        
    if show_fig:
        plt.show()
    else:
        plt.close()
        
    sns.reset_orig()
    

def plot_losses(losses, fig_path=os.getcwd(), show_fig=False, save_fig=False):
    """Plotting the losses for learning the player parameters for each player on one plot.
    
    :param losses: (num_iterations x num_players) numpy array of player losses for each iteration.
    :params fig_path, show_fig, savefig: path to save fig, whether to show, whether to save.    
    """

    loss_1 = losses[:, 0]
    loss_2 = losses[:, 1]

    sns.set(rc={'text.usetex' : True}, font_scale=1.0)
    sns.set_style("whitegrid", 
                 {'font.family':['serif'],
                  'font.serif':['Times New Roman']})

    fig = plt.figure(1,figsize=(6.5,6))
    fig.clf()
    ax = fig.add_subplot(111)

    lw = 5.5
    fs1 = 22
    fs2 = 24
    fs3 = 26

    ax.plot(loss_1, label=r'$l(\theta_1)$', color=xkcd['tomato red'], linestyle='-', lw=lw)
    ax.plot(loss_2, label=r'$l(\theta_2)$', color=xkcd['muted blue'], linestyle='--', lw=lw)

    ax.set_title('Loss', fontsize=fs2)
    ax.set_xlabel('iteration', fontsize=fs2)
    ax.set_ylabel(r'$\ell(\theta_i)$', fontsize=fs3, rotation=0, labelpad=30)

    lgd = ax.legend(loc='best', fancybox=True, fontsize=fs1)
    
    ax.tick_params(labelsize=fs2)
    ax.set_xlim([-20, len(losses)+20])
    fig.canvas.draw()
    
    if save_fig:
        plt.savefig(os.path.join(fig_path, 'loss.pdf'), bbox_extra_artists=(lgd,), 
                    bbox_inches='tight', dpi=300)
    
    if show_fig:
        plt.show()
    else:
        plt.close()

    sns.reset_orig()


def plot_player_parameters(estimation_parameters, player_number, fig_path=os.getcwd(), 
                           show_fig=False, save_fig=False):
    """Plotting the estimation parameters for a single player.

    :param estimation_parameters: List of player estimation parameters for each 
    iteration, each item is a numpy array of number of estimation params x number of players.
    :params fig_path, show_fig, savefig: path to save fig, whether to show, whether to save.   
    """

    estimation = np.vstack([param[:, player_number] for param in estimation_parameters])

    _all_estimation = np.array(estimation_parameters)
    max_estimation = int(np.ceil(_all_estimation.max()))
    min_estimation = int(np.floor(_all_estimation.min()))

    sns.set(rc={'text.usetex' : True}, font_scale=1.0)
    sns.set_style("whitegrid", 
                 {'font.family':['serif'],
                  'font.serif':['Times New Roman']})

    fig = plt.figure(1,figsize=(6.5,6))
    fig.clf()
    ax = fig.add_subplot(111)

    lw = 5.5
    fs1 = 22
    fs2 = 24
    fs3 = 26

    for i in xrange(estimation.shape[1]):
        ax.plot(estimation[:, i], label=r'$\hat{\theta}_{%d,%d}$' % (player_number+1, (i+1)), lw=lw, linestyle='-')
                
    ax.set_title('Agent %d Estimation Parameters' % (player_number+1), fontsize=fs2)
    ax.set_xlabel('iteration', fontsize=fs2)
    ax.set_ylabel(r'$\hat{\theta}_{%d,j}$' % (player_number+1), fontsize=fs3, rotation=0, labelpad=30)
    
    if estimation.shape[1] > 3:
        lgd = ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fancybox=True, ncol=1, fontsize=fs1)
    else:
        lgd = ax.legend(loc='best', fancybox=True, ncol=1, fontsize=fs1)

    ax.tick_params(labelsize=fs2)
    ax.set_xlim([-20, len(estimation)+20])
    ax.set_ylim([min_estimation-_all_estimation.ptp()/5., max_estimation+_all_estimation.ptp()/5.])

    fig.canvas.draw()

    if save_fig:
        plt.savefig(os.path.join(fig_path, 'estimation_player' + str(player_number+1) + '.pdf'), 
                    bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
        
    if show_fig:
        plt.show()
    else:
        plt.close()
        
    sns.reset_orig()


def plot_estimation_parameters(estimation_parameters, fig_path=os.getcwd(), 
                               show_fig=False, save_fig=False):
    """Plotting the estimation parameters for each player on subplots.

    :param estimation_parameters: List of player estimation parameters for each 
    iteration, each item is a numpy array of number of estimation params x number of players.
    :params fig_path, show_fig, savefig: path to save fig, whether to show, whether to save.   
    """

    estimation_1 = np.vstack([param[:, 0] for param in estimation_parameters])
    estimation_2 = np.vstack([param[:, 1] for param in estimation_parameters])

    sns.set(rc={'text.usetex' : True}, font_scale=1.0)
    sns.set_style("whitegrid", 
                 {'font.family':['serif'],
                  'font.serif':['Times New Roman']})

    lw = 5.5
    fs1 = 22
    fs2 = 24
    fs3 = 26

    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(13,6))
    fig.clf()

    ax1 = plt.subplot(1,2,1)
    for i in xrange(estimation_1.shape[1]):
        ax1.plot(estimation_1[:, i], label=r'$\hat{\theta}_{1,%d}$' % (i+1), lw=lw)
                
        ax1.set_title('Agent 1 Estimated Parameters', fontsize=fs2)
        ax1.set_xlabel('iteration', fontsize=fs2)
        ax1.set_ylabel(r'$\hat{\theta}_{1,j}$', fontsize=fs3, labelpad=30)
        
        if estimation_1.shape[1] > 2:
            lgd = ax1.legend(bbox_to_anchor=(.5, -.1), loc='upper center', fancybox=True, ncol=2, fontsize=fs1)
        else:
            lgd = ax1.legend(loc='best', fancybox=True, ncol=1, fontsize=fs1)

        ax1.tick_params(labelsize=fs2)
        ax1.set_xlim([-20, len(estimation_1)+20])
        
    ax2 = plt.subplot(1,2,2, sharey=ax1)
    for i in xrange(estimation_2.shape[1]):
        ax2.plot(estimation_2[:, i], label=r'$\hat{\theta}_{2,%d}$' % (i+1), lw=lw)
        
        ax2.set_title('Agent 2 Estimated Parameters', fontsize=fs2)
        ax2.set_xlabel('iteration', fontsize=fs2)
        ax2.set_ylabel(r'$\hat{\theta}_{2,j}$', fontsize=fs3, rotation=0, labelpad=30)
        

        if estimation_2.shape[1] > 2:
            lgd = ax2.legend(bbox_to_anchor=(.5, -.1), loc='upper center', fancybox=True, ncol=2, fontsize=fs1)
        else:
            lgd = ax2.legend(loc='best', fancybox=True, ncol=1, fontsize=fs1)
        
        ax2.tick_params(labelsize=fs2)
        ax2.set_xlim([-20, len(estimation_2)+20])

    plt.tight_layout()
    fig.canvas.draw()

    if save_fig:
        plt.savefig(os.path.join(fig_path, 'estimation_params.pdf'), 
                    bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
        
    if show_fig:
        plt.show()
    else:
        plt.close()
        
    sns.reset_orig()


def plot_estimation_parameters_together(estimation_parameters, fig_path=os.getcwd(), 
                                        show_fig=False, save_fig=False):
    """Plotting the estimation parameters for each player all on one plot.

    :param estimation_parameters: List of player estimation parameters for each 
    iteration, each item is a numpy array of number of estimation params x number of players.
    :params fig_path, show_fig, savefig: path to save fig, whether to show, whether to save.   
    """

    estimation_1 = np.vstack([param[:, 0] for param in estimation_parameters])
    estimation_2 = np.vstack([param[:, 1] for param in estimation_parameters])

    sns.set(rc={'text.usetex' : True}, font_scale=1.0)
    sns.set_style("whitegrid", 
                 {'font.family':['serif'],
                  'font.serif':['Times New Roman']})

    lw = 5.5
    fs1 = 22
    fs2 = 24
    fs3 = 26

    fig = plt.figure(1,figsize=(6.5,6))
    fig.clf()
    ax = fig.add_subplot(111)

    for i in xrange(estimation_1.shape[1]):
        ax.plot(estimation_1[:, i], label=r'$\hat{\theta}_{1,%d}$' % (i+1), lw=lw, linestyle='-')
        
    for i in xrange(estimation_2.shape[1]):
        ax.plot(estimation_2[:, i], label=r'$\hat{\theta}_{2,%d}$' % (i+1), lw=lw, linestyle='--')
        
    ax.set_title('Estimated Agent Parameters', fontsize=fs2)
    ax.set_xlabel('iteration', fontsize=fs2)
    ax.set_ylabel(r'$\hat{\theta}_{i,j}$', fontsize=fs3, rotation=0, labelpad=30)
    
    if estimation_1.shape[1] > 3:
        lgd = ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fancybox=True, ncol=2, fontsize=fs1)
    else:
        lgd = ax.legend(loc='best', fancybox=True, ncol=2, fontsize=fs1)
    
    ax.tick_params(labelsize=fs2)
    ax.set_xlim([-20, len(estimation_2)+20])
        
    fig.canvas.draw()

    if save_fig:
        plt.savefig(os.path.join(fig_path, 'estimation_params_together.pdf'), 
                    bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
        
    if show_fig:
        plt.show()
    else:
        plt.close()
        
    sns.reset_orig()


def plot_player_incentives(incentive_parameters, player_number, fig_path=os.getcwd(), 
                           show_fig=False, save_fig=False):
    """Plotting the incentive parameters for a single player.

    :param incentive_parameters: List of player incentive parameters for each 
    iteration, each item is a numpy array of number of incentives x number of players.
    :param player_number: Integer index of player starting at 0 to plot the incentives for.
    :params fig_path, show_fig, savefig: path to save fig, whether to show, whether to save.   
    """

    incentive = np.vstack([param[:, player_number] for param in incentive_parameters])

    _all_incentive = np.array(incentive_parameters).flatten()
    max_incentive = int(np.ceil(_all_incentive.max()))
    min_incentive = int(np.floor(_all_incentive.min()))

    sns.set(rc={'text.usetex' : True}, font_scale=1.0)
    sns.set_style("whitegrid", 
                 {'font.family':['serif'],
                  'font.serif':['Times New Roman']})

    fig = plt.figure(1,figsize=(6.5,6))
    fig.clf()
    ax = fig.add_subplot(111)

    lw = 5.5
    fs1 = 22
    fs2 = 24
    fs3 = 26

    for i in xrange(incentive.shape[1]):
        ax.plot(incentive[:, i], label=r'$\alpha_{%d,%d}$' % (player_number+1, (i+1)), lw=lw, linestyle='-')
                
    ax.set_title('Agent %d Incentive Parameters' % (player_number+1), fontsize=fs2)
    ax.set_xlabel('iteration', fontsize=fs2)
    ax.set_ylabel(r'$\alpha_{%d,j}$' % (player_number+1), fontsize=fs3, rotation=0, labelpad=30)
    
    if incentive.shape[1] > 3:
        lgd = ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fancybox=True, ncol=1, fontsize=fs1)
    else:
        lgd = ax.legend(loc='best', fancybox=True, ncol=1, fontsize=fs1)

    ax.tick_params(labelsize=fs2)
    ax.set_xlim([-20, len(incentive)+20])
    ax.set_ylim([min_incentive-_all_incentive.ptp()/5., max_incentive+_all_incentive.ptp()/5.])

    fig.canvas.draw()

    if save_fig:
        plt.savefig(os.path.join(fig_path, 'incentive_player' + str(player_number+1) + '.pdf'), 
                    bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
        
    if show_fig:
        plt.show()
    else:
        plt.close()
        
    sns.reset_orig()


def plot_incentive_parameters(incentive_parameters, fig_path=os.getcwd(), 
                              show_fig=False, save_fig=False):
    """Plotting the incentive parameters for each player all on one plot.

    :param incentive_parameters: List of player incentive parameters for each 
    iteration, each item is a numpy array of number of incentives x number of players.
    :params fig_path, show_fig, savefig: path to save fig, whether to show, whether to save.   
    """

    incentive_1 = np.vstack([param[:, 0] for param in incentive_parameters])
    incentive_2 = np.vstack([param[:, 1] for param in incentive_parameters])

    max_incentive = int(round(max(incentive_1.max(), incentive_2.max()), -1))
    min_incentive = int(round(min(incentive_1.min(), incentive_2.min()), -1))

    sns.set(rc={'text.usetex' : True}, font_scale=1.0)
    sns.set_style("whitegrid", 
                 {'font.family':['serif'],
                  'font.serif':['Times New Roman']})

    lw = 5.5
    fs1 = 22
    fs2 = 24
    fs3 = 26

    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(13,6))
    fig.clf()

    ax1 = plt.subplot(1,2,1)
    for i in xrange(incentive_1.shape[1]):
        ax1.plot(incentive_1[:, i], label=r'$\alpha_{1,%d}$' % (i+1), lw=lw)
                
        ax1.set_title('Agent 1 Incentive Parameters', fontsize=fs2)
        ax1.set_xlabel('iteration', fontsize=fs2)
        ax1.set_ylabel(r'$\alpha_{1,j}$', fontsize=fs3, rotation=0, labelpad=30)

        if incentive_1.shape[1] > 2:
            lgd = ax1.legend(bbox_to_anchor=(.5, -.1), loc='upper center', 
                             fancybox=True, ncol=2, fontsize=fs1)
        else:
            lgd = ax1.legend(loc='best', fancybox=True, ncol=1, fontsize=fs1)
        
        ax1.tick_params(labelsize=fs2)
        ax1.set_xlim([-20, len(incentive_1)+20])

    ax2 = plt.subplot(1,2,2, sharey=ax1)
    for i in xrange(incentive_2.shape[1]):
        ax2.plot(incentive_2[:, i], label=r'$\alpha_{2,%d}$' % (i+1), lw=lw)
                
        ax2.set_title('Agent 2 Incentive Parameters', fontsize=fs2)
        ax2.set_xlabel('iteration', fontsize=fs2)
        ax2.set_ylabel(r'$\alpha_{2,j}$', fontsize=fs3, rotation=0, labelpad=30)
        
        if incentive_2.shape[1] > 2:
            lgd = ax2.legend(bbox_to_anchor=(.5, -.1), loc='upper center', 
                             fancybox=True, ncol=2, fontsize=fs1)
        else:
            lgd = ax2.legend(loc='best', fancybox=True, ncol=1, fontsize=fs1)

        ax2.tick_params(labelsize=fs2)
        ax2.set_xlim([-20, len(incentive_2)+20])

    plt.tight_layout()
    fig.canvas.draw()

    if save_fig:
        plt.savefig(os.path.join(fig_path, 'incentive_params.pdf'), bbox_extra_artists=(lgd,), 
                    bbox_inches='tight', dpi=300)
        
    if show_fig:
        plt.show()
    else:
        plt.close()
        
    sns.reset_orig()


def plot_incentive_parameters_together(incentive_parameters, fig_path=os.getcwd(), 
                                        show_fig=False, save_fig=False):
    """Plotting the incentive parameters for each player all on one plot.

    :param incentive_parameters: List of player incentive parameters for each 
    iteration, each item is a numpy array of number of incentives x number of players.
    :params fig_path, show_fig, savefig: path to save fig, whether to show, whether to save.   
    """

    incentive_1 = np.vstack([param[:, 0] for param in incentive_parameters])
    incentive_2 = np.vstack([param[:, 1] for param in incentive_parameters])

    sns.set(rc={'text.usetex' : True}, font_scale=1.0)
    sns.set_style("whitegrid", 
                 {'font.family':['serif'],
                  'font.serif':['Times New Roman']})

    lw = 5.5
    fs1 = 22
    fs2 = 24
    fs3 = 26

    fig = plt.figure(1,figsize=(6.5,6))
    fig.clf()
    ax = fig.add_subplot(111)

    for i in xrange(incentive_1.shape[1]):
        ax.plot(incentive_1[:, i], label=r'$\alpha_{1,%d}$' % (i+1), lw=lw, linestyle='-')
        
    for i in xrange(incentive_2.shape[1]):
        ax.plot(incentive_2[:, i], label=r'$\alpha_{2,%d}$' % (i+1), lw=lw, linestyle='--')
        
    ax.set_title('Agent Incentive Parameters', fontsize=fs2)
    ax.set_xlabel('iteration', fontsize=fs2)
    ax.set_ylabel(r'$\alpha_{i,j}$', fontsize=fs3, rotation=0, labelpad=30)
    
    if incentive_1.shape[1] > 3:
        lgd = ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fancybox=True, ncol=2, fontsize=fs1)
    else:
        lgd = ax.legend(loc='best', fancybox=True, ncol=2, fontsize=fs1)
    
    ax.tick_params(labelsize=fs2)
    ax.set_xlim([-20, len(incentive_2)+20])
        
    fig.canvas.draw()

    if save_fig:
        plt.savefig(os.path.join(fig_path, 'incentive_params_together.pdf'), 
                    bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
        
    if show_fig:
        plt.show()
    else:
        plt.close()
        
    sns.reset_orig()



def plot_cost(learn_values, incentive_values, 
              fig_path=os.getcwd(), show_fig=False, save_fig=False):
    """Plotting the estimation and incentive portions of the cost function separately.
    
    :param learn_values: (num_iterations x num_players) numpy array of Phi(x)\hat{\theta} for each player.
    :param incentive_values: (num_iterations x num_players) numpy array of Psi(x)\alpha for each player.
    :params fig_path, show_fig, savefig: path to save fig, whether to show, whether to save.
    """

    learn1 = learn_values[:, 0]
    learn2 = learn_values[:, 1]
    
    incentive1 = incentive_values[:, 0]
    incentive2 = incentive_values[:, 1]

    sns.set(rc={'text.usetex' : True}, font_scale=1.0)
    sns.set_style("whitegrid", 
                 {'font.family':['serif'],
                  'font.serif':['Times New Roman']})
    
    fig = plt.figure(1,figsize=(6.5,6))
    fig.clf()
    ax = fig.add_subplot(111)

    lw = 5.5
    lw_ = 8
    fs1 = 22
    fs2 = 24
    fs3 = 26

    ax.plot(learn1, color=xkcd['reddish orange'], linestyle='-', lw=lw)
    line1 = lines.Line2D([], [], lw=lw, linestyle="-", color=xkcd['reddish orange'])

    ax.plot(incentive1, color=xkcd['reddish orange'], linestyle='-', lw=lw)
    ax.plot(incentive1, color=xkcd['steel grey'], linestyle=':', lw=lw)
    dotted_line1 = lines.Line2D([], [], lw=lw, linestyle=":", dashes=(10, 1), color=xkcd['reddish orange'])
    dotted_line2 = lines.Line2D([], [], lw=lw, linestyle="-", dashes=(5, 4), color=xkcd['steel grey'])

    ax.plot(learn1 + incentive1, color=xkcd['tomato red'], linestyle='-', lw=lw)
    ax.plot(learn1 + incentive1, color=xkcd['black'], linestyle='--', lw=lw)
    dotted_line3 = lines.Line2D([], [], lw=lw, linestyle="--", dashes=(10, 1), color=xkcd['tomato red'])
    dotted_line4 = lines.Line2D([], [], lw=lw, linestyle="-", dashes=(5, 4), color=xkcd['black'])

    ax.plot(learn2, color=xkcd['cobalt'], linestyle='-', lw=lw)
    line2 = lines.Line2D([], [], lw=lw, linestyle="-", color=xkcd['cobalt'])

    ax.plot(incentive2, color=xkcd['cobalt'], linestyle='-', lw=lw)
    ax.plot(incentive2, color=xkcd['steel grey'], linestyle=':', lw=lw)
    dotted_line5 = lines.Line2D([], [], lw=lw, linestyle=":", dashes=(10, 1), color=xkcd['cobalt'])
    dotted_line6 = lines.Line2D([], [], lw=lw, linestyle="-", dashes=(5, 4), color=xkcd['steel grey'])

    ax.plot(learn2 + incentive2, color=xkcd['muted blue'], linestyle='-', lw=lw)
    ax.plot(learn2 + incentive2, color=xkcd['black'], linestyle='--', lw=lw)
    dotted_line7 = lines.Line2D([], [], lw=lw, linestyle="--", dashes=(10, 1), color=xkcd['muted blue'])
    dotted_line8 = lines.Line2D([], [], lw=lw, linestyle="-", dashes=(5, 4), color=xkcd['black'])

    ax.set_title('', fontsize=fs2)
    ax.set_xlabel('iteration', fontsize=fs2)
    ax.set_ylabel(r'Utility Component', fontsize=fs3, rotation='vertical')

    lgd = ax.legend([(line1, line1), (dotted_line1, dotted_line2), (dotted_line3, dotted_line4), 
                     (line2, line2), (dotted_line5, dotted_line6), (dotted_line7, dotted_line8)], 
                    [r'$\langle \Phi_1(x), \hat{\theta}_1 \rangle + \nu$', 
                     r'$\langle \Psi_1(x), \alpha_1 \rangle$', 
                     r'$\langle \Phi_1(x), \hat{\theta}_1 \rangle + \nu + \langle \Psi_1(x), \alpha_1 \rangle$', 
                     r'$\langle \Phi_2(x), \hat{\theta}_2 \rangle + \nu$', 
                     r'$\langle \Psi_2(x), \alpha_2 \rangle$', 
                     r'$\langle \Phi_2(x), \hat{\theta}_2 \rangle + \nu + \langle \Psi_2(x), \alpha_2 \rangle$'],
                    bbox_to_anchor=(1, 1), loc='upper left', fancybox=True, ncol=1, fontsize=fs1)

    ax.tick_params(labelsize=fs2)
    ax.set_xlim([-20, len(learn_values)+20])
    fig.canvas.draw()
    
    if save_fig:
        plt.savefig(os.path.join(fig_path, 'cost.pdf'), bbox_extra_artists=(lgd,), 
                    bbox_inches='tight', dpi=300)
        
    if show_fig:
        plt.show()
    else:
        plt.close()
        
    sns.reset_orig()

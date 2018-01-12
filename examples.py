import os
import numpy as np
import utils
import online_ulid
from seaborn import xkcd_rgb as xkcd


def bertrand_example(play_type, incentive, marginal_revenue_type='nonlinear', 
                     history_length=1, planner_type=None, gp_step=.5, 
                     min_iter=50, max_iter=100, fig_path=os.getcwd(), 
                     show_fig=False, save_fig=True):
    """Bertrand example with online utility learning and incentive design.

    :param play_type: String indicating play type of the agents. Options are 
    gradient play 'gp', best response 'br', and fictitious play 'fp'.
    :param incentive: Bool indicating whether to use incentive or do estimation only.
    :param marginal_revenue_type: String indicating either 'linear' or 'nonlinear'
    marginal revenue function.
    :param history_length: Integer length of history to use in myopic update.
    :param min_iter: Minimum number of iterations to run the algorithm for.
    :param max_iter: Maximum number of iterations to run the algorithm for.
    :param fig_path: Path to save figures.
    :param show_fig: Bool indicating whether to save figure.
    :param save_fig: Bool indicating whether to show the figure.
    """

    np.random.seed(10)

    # Initialization of returns.
    if incentive:
        x_true = np.array([[1., 1.]])
    else:
        x_true = np.array([[1., 1.]])

    # Initialization of the desired returns.
    if incentive:
        x_desired = np.array([[14., 8.]])
    else:
        x_desired = None

    # Constraint on the upper bound of player return. 
    x_bar = 10000.

    # True parameters for the demand functions.
    if planner_type is None:
        theta_true = np.array([[-1.2, .3], [-.5, -1.], [7.5, 1.5]])
    elif planner_type == 'gp':
        theta_true = np.array([[-1.2, .3], [-.5, -1.]])

    # This parameter we assume is known.
    theta_known = np.array([[1., 1.]])

    # Initialization of the basis functions for the estimation.
    if planner_type is None:
        est_basis_functions = []
        est_basis_functions.append(lambda x, i: np.exp(-1e-2*((x[0, i])**2)))
        est_basis_functions.append(lambda x, i: np.exp(-1e-2*((x[0, 0] - x[0, 1])**2)))
        est_basis_functions.append(lambda x, i: np.exp(-1e-2*((x[0, 1])**2)) if i == 0 
                                                else np.exp(-1e-2*((x[0, 0])**2)))
    elif planner_type == 'gp':
        est_basis_functions = []
        est_basis_functions.append(lambda x, i: 2*x[0, i] if i == 0 else x[0, 0])
        est_basis_functions.append(lambda x, i: x[0, 1] if i == 0 else 2*x[0, i])

    # Initialization of the basis functions for the incentives.
    if incentive:
        inc_basis_functions = []
        inc_deriv_basis_functions = []

        inc_basis_functions.append(lambda x, x_d, i: np.exp(-1e-2*((x[0, i] - x_d[0, i])**2)))
        inc_deriv_basis_functions.append(lambda x, x_d, i: -1e-2*(x[0, i] - x_d[0, i])
                                                            *np.exp(-1e-2*((x[0, i] - x_d[0, i])**2)))

        inc_basis_functions.append(lambda x, x_d, i: np.exp(-1e-2*((x[0, i] + x_d[0, i])**2)))
        inc_deriv_basis_functions.append(lambda x, x_d, i: -1e-2*(x[0, i] + x_d[0, i])
                                                            *np.exp(-1e-2*((x[0, i] + x_d[0, i])**2)))

        inc_basis_functions.append(lambda x, x_d, i: np.exp(-1e-2*((x[0, i])**2)))
        inc_deriv_basis_functions.append(lambda x, x_d, i: -1e-2*(x[0, i])*np.exp(-1e-2*((x[0, i])**2)))
    else:
        inc_basis_functions = []
        inc_deriv_basis_functions = []

    # Firm knowledge parameters.
    if planner_type is None:
        mu = 5
        sigma = .5        
    elif planner_type == 'gp':
        mu = 5
        sigma = .5

    # Initializing the marginal revenue function.
    if marginal_revenue_type == 'linear':
        if planner_type is None:
            marginal_revenue = online_ulid.linear_marginal_revenue
            best_response = online_ulid.best_response_linear
        elif planner_type == 'gp':
            marginal_revenue = online_ulid.linear_marginal_revenue_
            best_response = online_ulid.best_response_linear
    elif marginal_revenue_type == 'nonlinear':
        marginal_revenue = online_ulid.nonlinear_marginal_revenue
        best_response = None 

    # Running main algorithm.
    results = online_ulid.online_ulid(x_true=x_true, x_desired=x_desired, x_bar=x_bar, 
                                      theta_true=theta_true, theta_known=theta_known, 
                                      est_basis_functions=est_basis_functions, 
                                      inc_basis_functions=inc_basis_functions, 
                                      inc_deriv_basis_functions=inc_deriv_basis_functions,
                                      marginal_revenue=marginal_revenue,
                                      best_response=best_response,
                                      mu=mu, sigma=sigma, 
                                      history_length=history_length, 
                                      play_type=play_type, 
                                      planner_type=planner_type, 
                                      gp_step=gp_step, min_iter=min_iter, 
                                      max_iter=max_iter)

    if incentive:
        if planner_type is None:
            true_agent_returns, estimated_agent_returns, excitations, losses, estimation_parameters, incentive_parameters, learn_values, incentive_values = results
        else:
            true_agent_returns, estimated_agent_returns, excitations, losses, estimation_parameters, incentive_parameters = results
    else:
        true_agent_returns, estimated_agent_returns, excitations, losses, estimation_parameters = results

    if planner_type is None:
        utils.plot_returns(true_agent_returns, estimated_agent_returns, x_desired,
                           fig_path=fig_path, show_fig=show_fig, save_fig=save_fig) 
    elif planner_type == 'gp':
        utils.plot_returns(true_agent_returns[:100], estimated_agent_returns[:100], x_desired,
                           fig_path=fig_path, show_fig=show_fig, save_fig=save_fig) 

    utils.plot_persistence_excitation(excitations, fig_path=fig_path, 
                                      show_fig=show_fig, save_fig=save_fig)
    utils.plot_losses(losses, fig_path=fig_path, show_fig=show_fig, save_fig=save_fig)
    utils.plot_estimation_parameters(estimation_parameters, fig_path=fig_path, 
                                     show_fig=show_fig, save_fig=save_fig)
    utils.plot_estimation_parameters_together(estimation_parameters, fig_path=fig_path, 
                                              show_fig=show_fig, save_fig=save_fig)
    utils.plot_player_parameters(estimation_parameters, player_number=0, 
                                 fig_path=fig_path, show_fig=show_fig, save_fig=save_fig)
    utils.plot_player_parameters(estimation_parameters, player_number=1, 
                                 fig_path=fig_path, show_fig=show_fig, save_fig=save_fig)
    if incentive:
        utils.plot_incentive_parameters(incentive_parameters, fig_path=fig_path, 
                                        show_fig=show_fig, save_fig=save_fig)
        utils.plot_incentive_parameters_together(incentive_parameters, fig_path=fig_path, 
                                                 show_fig=show_fig, save_fig=save_fig)
        utils.plot_player_incentives(incentive_parameters, player_number=0, 
                                     fig_path=fig_path, show_fig=show_fig, save_fig=save_fig)
        utils.plot_player_incentives(incentive_parameters, player_number=1, 
                                     fig_path=fig_path, show_fig=show_fig, save_fig=save_fig)

    if planner_type is None and incentive:
        utils.plot_cost(learn_values, incentive_values, fig_path=fig_path, 
                        show_fig=show_fig, save_fig=save_fig)

    if planner_type is None:
        if incentive:
            estimation_error = np.linalg.norm(true_agent_returns - estimated_agent_returns, axis=1)
            incentive_error = np.linalg.norm(true_agent_returns - x_desired, axis=1)
            return estimation_error, incentive_error
        else:
            estimation_error = np.linalg.norm(true_agent_returns - estimated_agent_returns, axis=1)
            return estimation_error
    elif planner_type == 'gp':
        color = xkcd['tomato red']

        estimation_error = np.linalg.norm(true_agent_returns - estimated_agent_returns, axis=1)
        utils.plot_estimation_error(estimation_errors=[estimation_error],  
                                    colors=[color], fig_path=fig_path, 
                                    show_fig=show_fig, save_fig=save_fig)

        parameter_error = np.linalg.norm(estimation_parameters - theta_true, axis=1)
        utils.plot_parameter_error(parameter_error=parameter_error,  
                                   fig_path=fig_path, show_fig=show_fig, 
                                   save_fig=save_fig)

        if incentive:
            incentive_error = np.linalg.norm(true_agent_returns - x_desired, axis=1)

            utils.plot_incentive_error(incentive_errors=[incentive_error],  
                                       colors=[color], fig_path=fig_path, 
                                       show_fig=show_fig, save_fig=save_fig)


def linear_estimation_examples():
    """Estimation on linear bertrand example using each agent update type."""

    play_type = 'gp'
    incentive = False
    marginal_revenue_type = 'linear'
    history_length = 1
    min_iter = 50
    max_iter = 100
    show_fig = False
    save_fig = True

    # Linear estimation using gradient play update for agents.
    fig_path = os.path.join(os.getcwd(), 'Figs', 'Linear_Estimation', 'GP')
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    gp_results = bertrand_example(play_type=play_type, incentive=incentive, 
                                  marginal_revenue_type=marginal_revenue_type, 
                                  history_length=history_length, min_iter=min_iter, 
                                  max_iter=max_iter, fig_path=fig_path, show_fig=show_fig, 
                                  save_fig=save_fig)

    # Linear estimation using best response update for agents.
    fig_path = os.path.join(os.getcwd(), 'Figs', 'Linear_Estimation', 'BR')
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)
        
    play_type = 'br'

    br_results = bertrand_example(play_type=play_type, incentive=incentive, 
                                  marginal_revenue_type=marginal_revenue_type, 
                                  history_length=history_length, min_iter=min_iter, 
                                  max_iter=max_iter, fig_path=fig_path, show_fig=show_fig, 
                                  save_fig=save_fig)

    # Linear estimation using fictitious play update for agents.
    fig_path = os.path.join(os.getcwd(), 'Figs', 'Linear_Estimation', 'FP')
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)
        
    play_type = 'fp'

    fp_results = bertrand_example(play_type=play_type, incentive=incentive, 
                                  marginal_revenue_type=marginal_revenue_type, 
                                  history_length=history_length, min_iter=min_iter, 
                                  max_iter=max_iter, fig_path=fig_path, show_fig=show_fig, 
                                  save_fig=save_fig)

    # Comparing algorithm performance.
    estimation_errors = [gp_results, br_results, fp_results]
    labels = ['Gradient Play', 'Best Response', 'Fictitious Play']
    colors = [xkcd['tomato red'], xkcd['muted blue'], xkcd['purple']]
    fig_path = os.path.join(os.getcwd(), 'Figs', 'Linear_Estimation')
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)
    utils.plot_estimation_error(estimation_errors=estimation_errors, labels=labels, 
                                colors=colors, fig_path=fig_path, show_fig=show_fig, 
                                save_fig=save_fig)


def nonlinear_estimation_examples():
    """Estimation on nonlinear bertrand example using each agent update type."""

    play_type = 'gp'
    incentive = False
    marginal_revenue_type = 'nonlinear'
    history_length = 1
    min_iter = 50
    max_iter = 100
    show_fig = False
    save_fig = True

    # Nonlinear estimation using gradient play update for agents.
    fig_path = os.path.join(os.getcwd(), 'Figs', 'Nonlinear_Estimation', 'GP')
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    gp_results = bertrand_example(play_type=play_type, incentive=incentive, 
                                  marginal_revenue_type=marginal_revenue_type, 
                                  history_length=history_length, min_iter=min_iter, 
                                  max_iter=max_iter, fig_path=fig_path, show_fig=show_fig, 
                                  save_fig=save_fig)

    estimation_errors = [gp_results]
    colors = [xkcd['purple']]
    fig_path = os.path.join(os.getcwd(), 'Figs', 'Nonlinear_Estimation')
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)
    utils.plot_estimation_error(estimation_errors=estimation_errors, 
                                colors=colors, fig_path=fig_path, 
                                show_fig=show_fig, save_fig=save_fig)


def linear_online_ulid_examples():
    """Online utility learning and incentive design on linear bertrand example 
    using each agent update type."""

    play_type = 'gp'
    incentive = True
    marginal_revenue_type = 'linear'
    history_length = 1
    min_iter = 50
    max_iter = 100
    show_fig = False
    save_fig = True

    # Linear ULID using gradient play update for agents.
    fig_path = os.path.join(os.getcwd(), 'Figs', 'Linear_ULID', 'GP')
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    gp_results = bertrand_example(play_type=play_type, incentive=incentive, 
                                  marginal_revenue_type=marginal_revenue_type, 
                                  history_length=history_length, min_iter=min_iter, 
                                  max_iter=max_iter, fig_path=fig_path, show_fig=show_fig, 
                                  save_fig=save_fig)

    # Linear ULID using best response update for agents.
    fig_path = os.path.join(os.getcwd(), 'Figs', 'Linear_ULID', 'BR')
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)
        
    play_type = 'br'

    br_results = bertrand_example(play_type=play_type, incentive=incentive, 
                                  marginal_revenue_type=marginal_revenue_type, 
                                  history_length=history_length, min_iter=min_iter, 
                                  max_iter=max_iter, fig_path=fig_path, show_fig=show_fig, 
                                  save_fig=save_fig)


    # Linear ULID using fictitious play update for agents.
    fig_path = os.path.join(os.getcwd(), 'Figs', 'Linear_ULID', 'FP')
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)
        
    play_type = 'fp'

    fp_results = bertrand_example(play_type=play_type, incentive=incentive, 
                                  marginal_revenue_type=marginal_revenue_type, 
                                  history_length=history_length, min_iter=min_iter, 
                                  max_iter=max_iter, fig_path=fig_path, show_fig=show_fig, 
                                  save_fig=save_fig)

    # Comparing algorithm performance.
    estimation_errors = [gp_results[0], br_results[0], fp_results[0]]
    incentive_errors = [gp_results[1], br_results[1], fp_results[1]]
    labels = ['Gradient Play', 'Best Response', 'Fictitious Play']
    colors = [xkcd['tomato red'], xkcd['muted blue'], xkcd['purple']]
    fig_path = os.path.join(os.getcwd(), 'Figs', 'Linear_ULID')
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)
    utils.plot_estimation_error(estimation_errors=estimation_errors, labels=labels, 
                                colors=colors, fig_path=fig_path, show_fig=show_fig, 
                                save_fig=save_fig)
    utils.plot_incentive_error(incentive_errors=incentive_errors, labels=labels, 
                                colors=colors, fig_path=fig_path, show_fig=show_fig, 
                                save_fig=save_fig)


def nonlinear_online_ulid_examples():
    """Online utility learning and incentive design on nonlinear bertrand 
    example using each agent update type."""

    play_type = 'gp'
    incentive = True
    marginal_revenue_type = 'nonlinear'
    history_length = 1
    min_iter = 50
    max_iter = 100
    show_fig = False
    save_fig = True

    # Nonlinear ULID using gradient play update for agents.
    fig_path = os.path.join(os.getcwd(), 'Figs', 'Nonlinear_ULID', 'GP')
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    gp_results = bertrand_example(play_type=play_type, incentive=incentive, 
                                  marginal_revenue_type=marginal_revenue_type, 
                                  history_length=history_length, min_iter=min_iter, 
                                  max_iter=max_iter, fig_path=fig_path, show_fig=show_fig, 
                                  save_fig=save_fig)

    estimation_errors = [gp_results[0]]
    incentive_errors = [gp_results[1]]
    colors = [xkcd['purple']]
    fig_path = os.path.join(os.getcwd(), 'Figs', 'Nonlinear_ULID')
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)
    utils.plot_estimation_error(estimation_errors=estimation_errors,  
                                colors=colors, fig_path=fig_path, 
                                show_fig=show_fig, save_fig=save_fig)
    utils.plot_incentive_error(incentive_errors=incentive_errors, 
                                colors=colors, fig_path=fig_path, 
                                show_fig=show_fig, save_fig=save_fig)


def true_gp_examples():
    """Bertrand example with the agents playing gp and the planner having knowledge of this.

    In these examples instead of the planner using an agnostic update with 
    various basis functions like in the other examples, the planner has knowledge
    the agents are playing gradient play. This example illustrates how in estimation
    and with incentive design as well the agents are pushed to the desired response
    and the planners parameter estimates converge to the true ones.
    """

    play_type = 'gp'
    incentive = False
    marginal_revenue_type = 'linear'
    history_length = 1
    planner_type = 'gp'
    gp_step = .55
    min_iter = 50
    max_iter = 5000
    show_fig = False
    save_fig = True

    # Linear estimation using gradient play update for agents and planner.
    fig_path = os.path.join(os.getcwd(), 'Figs', 'True_GP', 'Linear_Estimation')
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    gp_results = bertrand_example(play_type=play_type, incentive=incentive, 
                                  marginal_revenue_type=marginal_revenue_type, 
                                  history_length=history_length, 
                                  planner_type=planner_type, 
                                  gp_step=gp_step, min_iter=min_iter, 
                                  max_iter=max_iter, fig_path=fig_path, 
                                  show_fig=show_fig, save_fig=save_fig)

    # Linear ULID using gradient play update for agents and planner.
    fig_path = os.path.join(os.getcwd(), 'Figs', 'True_GP', 'Linear_ULID')
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    incentive = True

    gp_results = bertrand_example(play_type=play_type, incentive=incentive, 
                                  marginal_revenue_type=marginal_revenue_type, 
                                  history_length=history_length, 
                                  planner_type=planner_type, 
                                  gp_step=gp_step, min_iter=min_iter, 
                                  max_iter=max_iter, fig_path=fig_path, 
                                  show_fig=show_fig, save_fig=save_fig)


def main():
    linear_estimation_examples()
    nonlinear_estimation_examples()
    linear_online_ulid_examples()
    nonlinear_online_ulid_examples()
    true_gp_examples()


if __name__ == "__main__":
    main()
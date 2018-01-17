from __future__ import print_function
import os
import numpy as np


def online_ulid(x_true, x_desired, x_bar, theta_true, theta_known, 
                est_basis_functions, inc_basis_functions=[], 
                inc_deriv_basis_functions=[], marginal_revenue=None,
                best_response=None, mu=5, sigma=1.5, history_length=1, 
                play_type='gp', planner_type=None, gp_step=.5,
                min_iter=50, max_iter=100):    
    """Online utility learning and incentive design for bertand example.

    :param x_true: (1 x num_players) numpy array of the initial true player returns.
    :param x_desired: (1 x num_players) numpy array of the desired player returns.
    :param x_bar: Upper bound for the agent returns.  
    :param theta_true: (3 x num_players) numpy array of the true player parameters.
    :param theta_known: (1 x num_players) numpy array of the known parameter for each player.
    :param est_basis_functions: List of basis functions to map return with for estimation.
    :param inc_basis_functions: List of basis functions to map return with for incentives.
    :param inc_deriv_basis_functions: List of derivative basis functions for step size computation.
    :param marginal_revenue: Function to compute marginal revenue for a player.
    :param best_response: Function to compute player equilibrium.
    :param mu: Mean for normal random variable for firm knowledge.
    :param sigma: Standard deviation for normal random variable for firm knowledge.
    :param history_length: Integer of the number of past responses to use as average.
    :param play_type: String indicating play type of the agents. Options are 
    gradient play 'gp', best response 'br', and fictitious play 'fp'.
    :param planner_type: Type of estimation and incentive update. If None this indicates
    to assume no knowledge of the true process. If gp, the estimation and incentive
    and updates will use gp updates.
    :param gp_step: Float step size to use for gradient play updates.
    :param min_iter: Minimum number of iterations to run the algorithm for.
    :param max_iter: Maximum number of iterations to run the algorithm for.

    :return true_agent_returns: (num_iterations x num_players) numpy array of true returns for each iteration.
    :return estimated_agent_returns: (num_iterations x num_players) numpy array of estimated returns for each iteration.
    :return excitations: (num_iterations x num_players) numpy array of norm squared of regression vectors for each iteration.
    :return losses: (num_iterations x num_players) numpy array of player losses for each iteration.
    :return estimation_parameters: List of player estimation parameters for each iteration.
    :return incentive_parameters: List of player incentive parameters for each iteration.
    :return learn_val: (num_iterations x num_players) numpy array containing 
    \Phi(x)\hat{\theta} for each player at each iteration.
    :return incentive_val: (num_iterations x num_players) numpy array containing
    \Psi(x)\alpha for each player at each iteration.
    """

    num_players = 2
    theta_hat = np.zeros((len(est_basis_functions), num_players))
    alpha = np.zeros((len(inc_basis_functions), num_players))

    if len(inc_basis_functions) == 0:
        incentive = False
    else:
        incentive = True

    if play_type not in ['gp', 'br', 'fp']:
        print('Invalid play type: defaulting to gradient play')
        play_type = 'gp'
    if planner_type not in [None, 'gp']:
        print('Invalid planner type: defaulting to agnostic planner type')
        planner_type = None

    true_agent_returns = [x_true]    
    estimated_agent_returns = [np.zeros(x_true.shape)]
    estimation_parameters = [theta_hat]
    incentive_parameters = [alpha]
    excitations = []
    losses = []
    if incentive:
        learn_vals = []
        incentive_vals = []
    done = False
    k = 0

    while k <= max_iter and done == False:
        nu = update_firm_knowledge(mu, sigma, num_players)

        x_true = np.vstack(true_agent_returns)[-history_length:].mean(axis=0, keepdims=True)

        x_true_update = update_true_return(marginal_revenue, best_response,
                                           inc_basis_functions, 
                                           inc_deriv_basis_functions,
                                           x_true, x_desired, alpha, theta_true, 
                                           theta_known, nu, x_bar, num_players, 
                                           incentive, play_type, planner_type, 
                                           gp_step, k)

        if incentive:
            params = update_estimated_return(est_basis_functions, inc_basis_functions,
                                             x_true, x_desired, alpha, theta_hat, 
                                             theta_known, nu, num_players, incentive, 
                                             planner_type, gp_step) 
            x_hat_update, learn_val, incentive_val = params 
        else:
            x_hat_update = update_estimated_return(est_basis_functions, inc_basis_functions,
                                                   x_true, x_desired, alpha, theta_hat, 
                                                   theta_known, nu, num_players, incentive, 
                                                   planner_type, gp_step) 

        xi_k = update_regression_vectors(est_basis_functions, x_true, 
                                         num_players, planner_type, gp_step)
        

        y_k_update = update_observations(inc_basis_functions, x_true, x_desired,
                                         x_true_update, alpha, theta_known, 
                                         nu, num_players, incentive, 
                                         planner_type, gp_step)
        
        loss = incur_loss(y_k_update, xi_k, theta_hat, num_players)
        
        theta_hat_update = update_player_params(y_k_update, xi_k, theta_hat, num_players)    

        # Check for parameter convergence in gradient play case.
        if planner_type == 'gp':
            param_err = np.linalg.norm(theta_true - theta_hat_update, axis=1)
            if param_err.max() < .01:
                done = True
    
        if incentive:
            alpha = update_incentive_params(est_basis_functions, inc_basis_functions, 
                                            x_true, x_desired, theta_hat_update, 
                                            theta_known, nu, num_players, 
                                            planner_type, gp_step)
            incentive_parameters.append(alpha)
        
        true_agent_returns.append(x_true_update)
        estimated_agent_returns.append(x_hat_update)
        losses.append(loss)
        estimation_parameters.append(theta_hat_update)
        excitations.append([np.linalg.norm(xi_k[:, i, None]**2) for i in xrange(xi_k.shape[1])])
        if incentive:
            learn_vals.append(learn_val)
            incentive_vals.append(incentive_val)

        theta_hat = theta_hat_update

        k += 1
                        
    true_agent_returns = np.vstack(true_agent_returns)
    estimated_agent_returns = np.vstack(estimated_agent_returns)
    losses = np.vstack(losses)
    excitations = np.vstack(excitations)
    if incentive:
        learn_vals = np.array(learn_vals)
        incentive_vals = np.array(incentive_vals)

    if incentive:
        results = (true_agent_returns, estimated_agent_returns, excitations, 
                   losses, estimation_parameters, incentive_parameters, 
                   learn_vals, incentive_vals)
    else:
        results = (true_agent_returns, estimated_agent_returns, excitations, 
                   losses, estimation_parameters)

    return results


def play_step(x_true, theta_true):
    """Compute the step size to use for the true process update in GP with no incentive.

    :param x_true: (1 x num_players) numpy array of the true player returns.
    :param theta_true: (3 x num_players) numpy array of the true player parameters.

    :return step: Float step size for the step size.
    """

    # Compute the hessian of the marginal revenue function.
    d2f1 = [1./x_true[0, 0] + theta_true[0, 0] + theta_true[0, 0], theta_true[0, 1]]
    d2f2 = [theta_true[1, 0], 1./x_true[0, 1] + theta_true[1, 1] + theta_true[1, 1]]

    hessian = np.vstack((d2f1, d2f2))
    lipschitz = np.linalg.norm(hessian, 2)
    step = .05/lipschitz

    return step


def play_step_inc(inc_deriv_basis_functions, x_true, theta_true, x_desired, alpha):
    """Compute the step size to use for the true process update with incentive in GP.

    :param inc_deriv_basis_functions: List of derivative basis functions for step size computation.
    :param x_true: (1 x num_players) numpy array of the true player returns.
    :param theta_true: (3 x num_players) numpy array of the true player parameters.
    :param x_desired: (1 x num_players) numpy array of the desired player returns.
    :param alpha: (num_inc_basis_functions x num_players) numpy array of incentive parameters.

    :return step: Float step size for gradient play update.
    """

    # Compute the hessian of the marginal revenue function.
    d2f1_marg = [1./x_true[0, 0] + theta_true[0, 0] + theta_true[0, 0], theta_true[0, 1]]
    d2f2_marg = [theta_true[1, 0], 1./x_true[0, 1] + theta_true[1, 1] + theta_true[1, 1]]
    hessian_marg = np.vstack((d2f1_marg, d2f2_marg))

    # Compute the hessian of the incentive function.
    d2f1_inc = [np.asscalar(np.dot(psi(inc_deriv_basis_functions, x_true, x_desired, 0).T, 
                                   alpha[:, 0, None])), 0.]
    d2f2_inc = [0., np.asscalar(np.dot(psi(inc_deriv_basis_functions, x_true, x_desired, 1).T, 
                                       alpha[:, 1, None]))]
    hessian_inc = np.vstack((d2f1_inc, d2f2_inc))

    hessian = hessian_marg + hessian_inc
    lipschitz = np.linalg.norm(hessian, 2)
    step = .25/lipschitz

    return step


def theta_step(xi_k, i):
    """Compute a stable step size to use for the loss function update.

    :param xi_k: (num_est_basis_functions x num_players) numpy array of regression vectors.

    :return step: Float step size for gradient descent.
    """

    step = .5/(1+np.linalg.norm(xi_k[:, i, None])**2)

    return step


def linear_marginal_revenue_(x_true, theta_true, theta_known, nu, i):
    """Computing the linear marginal revenue for the player returns.
    
    This corresponds to the update equation of:
    M_i^*(x_1, x_2, nu_i, \theta_i^*) = \theta_{i1}^*x_1 + \theta_{i2}^*x_2 
                                         + w_{i}^*nu_i + \theta_{i3}^* + \theta_{ii}^*x_i.
    
    :param x_true: (1 x num_players) numpy array of player returns.
    :param theta_true: (3 x num_players) numpy array of the true player parameters.
    :param nu: (1 x num_players) numpy array representing firm specific knowledge.
    :param i: Player index the target vector is being calculated for.

    :return marg_rev: Marginal revenue value for player i.
    """
    
    params = np.array([x_true[:, 0], x_true[:, 1]])
    marg_rev = np.dot(theta_true[:, i, None].T, params) + theta_known[0, i]*nu[0, i] \
               + theta_true[i, i]*x_true[0, i]
    
    return marg_rev


def linear_marginal_revenue(x_true, theta_true, theta_known, nu, i):
    """Computing the linear marginal revenue for the player returns.
    
    This corresponds to the update equation of:
    M_i^*(x_1, x_2, nu_i, \theta_i^*) = \theta_{i1}^*x_1 + \theta_{i2}^*x_2 
                                         + w_{i}^*nu_i + \theta_{i3}^* + \theta_{ii}^*x_i.
    
    :param x_true: (1 x num_players) numpy array of player returns.
    :param theta_true: (3 x num_players) numpy array of the true player parameters.
    :param nu: (1 x num_players) numpy array representing firm specific knowledge.
    :param i: Player index the target vector is being calculated for.

    :return marg_rev: Marginal revenue value for player i.
    """
    
    params = np.array([x_true[:, 0], x_true[:, 1], [1]])
    marg_rev = np.dot(theta_true[:, i, None].T, params) + theta_known[0, i]*nu[0, i] \
               + theta_true[i, i]*x_true[0, i]
    
    return marg_rev


def best_response_linear(inc_basis_functions, x_desired, alpha, theta_true, 
                         theta_known, nu, num_players, incentive, play_type, 
                         tol=1e-2, max_iter=25):
    """Compute best response or fictitious play using a linear marginal revenue function.

    :param inc_basis_functions: List of basis functions to map return with for incentives.
    :param x_desired: (1 x num_players) numpy array of the desired player returns.
    :param alpha: (num_inc_basis_functions x num_players) numpy array of incentive parameters.
    :param theta_true: (3 x num_players) numpy array of the true player parameters.
    :param nu: (1 x num_players) of firm knowledge parameters.
    :param num_players: Number of agents in the system.
    :param incentive: Bool indicating whether to use incentives.
    :param play_type: String indicating play type of the agents. Options are 
     best response 'br', and fictitious play 'fp'.
    :param tol: Tolerance for convergence to equilibrium.
    :param max_iter: Maximum number of iterations to run best response for.
    """

    agent_returns = [np.random.rand(1, num_players)]

    for i in xrange(max_iter):
        if play_type == 'br':
            x = np.vstack(agent_returns)[-1:].mean(axis=0, keepdims=True)
        elif play_type == 'fp':
            # In FP use average of other players past responses.
            x = np.vstack(agent_returns)[:].mean(axis=0, keepdims=True)

        x_old = agent_returns[-1].copy()

        if incentive:
            inc_term = np.asscalar(np.dot(psi(inc_basis_functions, agent_returns[-1], x_desired, 0).T, 
                                          alpha[:, 0, None]))
            x1 = -(theta_true[1, 0]*x[0, 1] + theta_known[0, 0]*nu[0, 0]
                  +theta_true[2, 0] + inc_term)/(2.*theta_true[0, 0])
        else:
            x1 = -(theta_true[1, 0]*x[0, 1] + theta_known[0, 0]*nu[0, 0]
                  +theta_true[2, 0])/(2.*theta_true[0, 0])
        
        if incentive:
            inc_term = np.asscalar(np.dot(psi(inc_basis_functions, agent_returns[-1], x_desired, 1).T, 
                                          alpha[:, 1, None]))
            x2 = -(theta_true[0, 1]*x[0, 0] + theta_known[0, 1]*nu[0, 1]
                   + theta_true[2, 1] + inc_term)/(2.*theta_true[1, 1])
        else:
            x2 = -(theta_true[0, 1]*x[0, 0] + theta_known[0, 1]*nu[0, 1]
                  +theta_true[2, 1])/(2.*theta_true[1, 1])
    
        if x1 < 0:
            x1 = 0.
        else:
            pass
        
        if x2 < 0:
            x2 = 0
        else:
            pass

        agent_returns.append(np.array([[x1, x2]]))

        diff = np.linalg.norm(agent_returns[-1] - x_old)

        if diff < tol:
            break
        
    return agent_returns[-1]


def nonlinear_marginal_revenue(x_true, theta_true, theta_known, nu, i):
    """Computing the nonlinear marginal revenue for the player returns.
    
    This corresponds to the update equation of:
    M_i^*(x_1, x_2, nu_i, \theta_i^*) = log(p_i) + \theta_{i1}^*x_1 + \theta_{i2}^*x_2 
                                         + w_{i}^*nu_i + \theta_{i3}^* + 1 + \theta_{ii}^*x_i.
    
    :param x_true: (1 x num_players) numpy array of player returns.
    :param theta_true: (3 x num_players) numpy array of the true player parameters.
    :param nu: (1 x num_players) numpy array representing firm specific knowledge.
    :param i: Player index the target vector is being calculated for.

    :return marg_rev: Marginal revenue value for player i.
    """
    
    params = np.array([x_true[:, 0], x_true[:, 1], [1]])
    marg_rev = np.log(x_true[0, i]) + np.dot(theta_true[:, i, None].T, params) \
               + theta_known[0, i]*nu[0, i] + 1 + theta_true[i, i]*x_true[0, i]
    
    return marg_rev


def update_firm_knowledge(mu, sigma, num_players):
    """Sampling from a normal distribution for firm knowledge parameters.

    :param mu: Mean for normal random variable.
    :param sigma: Standard deviation for normal random variable.
    :param num_players: Number of agents in the system.

    :return nu: (1 x num_players) numpy array of firm knowledge parameters.
    """
    
    firm_knowledge = np.random.normal(mu, sigma)
    
    # Setting to the idealized case that this parameter is the same for each player.
    nu = np.array([[firm_knowledge for i in xrange(num_players)]])
    
    return nu


def update_true_return(marginal_revenue, best_response, inc_basis_functions, 
                       inc_deriv_basis_functions, x_true, x_desired, alpha, 
                       theta_true, theta_known, nu, x_bar, num_players, incentive, 
                       play_type, planner_type, gp_step, k):
    """Updating the true process generating returns for the players with GP, BR, or FP.

    :param marginal_revenue: Function to compute marginal revenue for a player.
    :param best_response: Function to find equilibrium using best response or fictitious play.
    :param inc_basis_functions: List of basis functions to map return with for incentives.
    :param inc_deriv_basis_functions: List of derivative basis functions for step size computation.
    :param x_true: (1 x num_players) numpy array of the true player returns.
    :param x_desired: (1 x num_players) numpy array of the desired player returns.
    :param alpha: (num_inc_basis_functions x num_players) numpy array of incentive parameters.
    :param theta_true: (3 x num_players) numpy array of the true player parameters.
    :param nu: (1 x num_players) of firm knowledge parameters.
    :param x_bar: Upper bound for the agent returns.  
    :param num_players: Number of agents in the system.
    :param incentive: Bool indicating whether to use incentives.
    :param play_type: String indicating play type of the agents. Options are 
    gradient play 'gp', best response 'br', and fictitious play 'fp'.
    :param planner_type: Estimation update method, if None update is agnostic, 
    if 'gp' update uses gradient play update.
    :param gp_step: Float step size to use for gradient play updates.
    """
    
    # Constraint on the upper bound of player return.
    x_true = x_true.clip(None, x_bar)

    if play_type == 'gp':
        x_true_update = np.zeros((1, num_players))

        if planner_type == 'gp':
            step = gp_step
        elif incentive:
            step = play_step_inc(inc_deriv_basis_functions, x_true, theta_true, 
                                 x_desired, alpha)
        else:
            step = play_step(x_true, theta_true)

    for i in xrange(num_players):
        if play_type == 'gp':
            if incentive:
                x_true_update[0, i] = x_true[0, i] \
                                      + step*(marginal_revenue(x_true, theta_true, theta_known, nu, i) 
                                              + np.dot(psi(inc_basis_functions, x_true, x_desired, i).T, 
                                                       alpha[:, i, None]))
            else:
                x_true_update[0, i] = x_true[0, i] \
                                      + step*marginal_revenue(x_true, theta_true, theta_known, nu, i)
        elif play_type == 'br' or play_type == 'fp':
            x_true_update = best_response(inc_basis_functions, x_desired, 
                                          alpha, theta_true, theta_known, nu, 
                                          num_players, incentive, play_type)
              
    return x_true_update


def update_estimated_return(est_basis_functions, inc_basis_functions, x_true, 
                            x_desired, alpha, theta_hat, theta_known, nu, 
                            num_players, incentive, planner_type, gp_step):
    """Update the players estimated return using a agnostic or gradient play myopic update.

    :param est_basis_functions: List of basis functions to map return with for estimation.
    :param inc_basis_functions: List of basis functions to map return with for incentives.
    :param x_true: (1 x num_players) numpy array of the true player returns.
    :param x_desired: (1 x num_players) numpy array of the desired player returns.
    :param alpha: (num_inc_basis_functions x num_players) numpy array of the incentive parameters.
    :param theta_hat: (num_est_basis_functions x num_players) numpy array of the estimated player parameters.
    :param theta_known: (1 x num_players) numpy array of the known parameter for each player.
    :param nu: (1 x num_players) numpy array of firm knowledge parameters.
    :param num_players: Number of agents in the system.
    :param incentive: Bool indicating whether to use incentives.
    :param planner_type: Estimation update method, if None update is agnostic, 
    if 'gp' update uses gradient play update.
    :param gp_step: Float step size to use for gradient play updates.

    :return x_hat_update: (1 x num_players) numpy array of the updated true player returns.
    :return learn_val: List containing value from \Phi(x)alpha for each player.
    :return incentive_val: List containing value from \Psi(x)\hat{\theta} for each player.
    """
    
    x_hat_update = np.zeros((1, num_players))

    if incentive:
        learn_val = []
        incentive_val = []

    for i in xrange(num_players):
        if incentive:
            marg_rev_term = np.dot(phi(est_basis_functions, x_true, i).T, 
                                   theta_hat[:, i, None]) \
                            + nu[0, i]*theta_known[0, i]
            inc_term = np.dot(psi(inc_basis_functions, x_true, x_desired, i).T, 
                              alpha[:, i, None])

            learn_val.append(np.asscalar(marg_rev_term))
            incentive_val.append(np.asscalar(inc_term))

            if planner_type is None:
                x_hat_update[0, i] = marg_rev_term + inc_term
            elif planner_type == 'gp':
                x_hat_update[0, i] = x_true[0, i] + gp_step*(marg_rev_term + inc_term)
        else:
            if planner_type is None:
                x_hat_update[0, i] = np.dot(phi(est_basis_functions, x_true, i).T, 
                                            theta_hat[:, i, None]) \
                                     + nu[0, i]*theta_known[0, i] 
            elif planner_type == 'gp':
                x_hat_update[0, i] = x_true[0, i] + gp_step*(np.dot(phi(est_basis_functions, x_true, i).T, 
                                                                   theta_hat[:, i, None]) 
                                                             + theta_known[0, i]*nu[0, i]) 
    if incentive:
        return x_hat_update, learn_val, incentive_val
    else:
        return x_hat_update


def update_regression_vectors(est_basis_functions, x_true, num_players, 
                              planner_type, gp_step):
    """Update the players regression vectors of the bertrand competition.
    
    :param est_basis_functions: List of basis functions to map return with for estimation.
    :param x_true: (1 x num_players) numpy array of the true player returns.
    :param num_players: Number of agents in the system.  
    :param planner_type: Update method, if None update is agnostic, if 'gp' update 
    uses gradient play update.
    :param gp_step: Float step size to use for gradient play updates.
    
    :return xi_k: (num_est_basis_functions x num_players) numpy array of regression vectors.
    """
    
    xi_k = np.zeros((len(est_basis_functions), num_players))

    for i in xrange(num_players):
        if planner_type == 'gp':
            xi_k[:, i, None] = gp_step*phi(est_basis_functions, x_true, i)
        else:
            xi_k[:, i, None] = phi(est_basis_functions, x_true, i)
    
    return xi_k

    
def update_observations(inc_basis_functions, x_true, x_desired, x_true_update, 
                        alpha, theta_known, nu, num_players, incentive, 
                        planner_type, gp_step):
    """Updating the observations of the bertrand competition.
    
    :param inc_basis_functions: List of basis functions to map return with for incentives.
    :param x_true: (1 x num_players) numpy array of the true player returns.
    :param x_desired: (1 x num_players) numpy array of the desired player returns.
    :param x_true_update: (1 x num_players) numpy array of the true player returns updated.
    :param alpha: (num_inc_basis_functions x num_players) numpy array of the incentive parameters.
    :param theta_known: (1 x num_players) numpy array of the known parameter for each player.
    :param nu: (1 x num_players) numpy array of firm knowledge parameters.
    :param num_players: Number of agents in the system.
    :param incentive: Bool indicating whether to use incentives.
    :param planner_type: Update method, if None update is agnostic, if 'gp' update 
    uses gradient play update.
    :param gp_step: Float step size to use for gradient play updates.
    
    :return y_k_update: (1 x num_players) numpy array of updated observations.
    """
    
    y_k_update = np.zeros((1, num_players))

    for i in xrange(num_players):
        if planner_type is None:
            if incentive:
                y_k_update[0, i] = x_true_update[0, i] - nu[0, i]*theta_known[0, i] \
                                   - np.dot(psi(inc_basis_functions, x_true, x_desired, i).T, 
                                            alpha[:, i, None]) 
            else:
                y_k_update[0, i] = x_true_update[0, i] - nu[0, i]*theta_known[0, i]  
        elif planner_type == 'gp':
            if incentive:
                y_k_update[0, i] = x_true_update[0, i] - x_true[0, i] \
                                     - gp_step*(nu[0, i]*theta_known[0, i]
                                                + np.dot(psi(inc_basis_functions, 
                                                             x_true, x_desired, i).T, 
                                                         alpha[:, i, None]))
            else:
                y_k_update[0, i] = x_true_update[0, i] - x_true[0, i] \
                                     - gp_step*(nu[0, i]*theta_known[0, i])

    return y_k_update


def phi(est_basis_functions, x_true, i):
    """Mapping of the agent return using provided basis functions for estimation.
    
    :param est_basis_functions: List of basis functions to map return with for estimation.
    :param x_true: (1 x num_players) numpy array of the true player returns.
    :param i: Player index the target vector is being calculated for.
    
    return phi: (num_est_basis_functions x 1) numpy array of mapped player returns.
    """
    
    phi = [basis(x_true, i) for basis in est_basis_functions]
    phi = np.vstack(phi)
    
    return phi


def psi(inc_basis_functions, x_true, x_desired, i):
    """Mapping of the agent return using provided basis functions for incentives.

    :param inc_basis_functions: List of basis functions to map return with for incentives.
    :param x_true: (1 x num_players) numpy array of the true player returns.
    :param x_desired: (1 x num_players) numpy array of the desired player returns.
    :param i: Player index the target vector is being calculated for.

    return psi: (num_inc_basis_functions x 1) numpy array of mapped player returns.
    """

    psi = [basis(x_true, x_desired, i) for basis in inc_basis_functions]
    psi = np.vstack(psi)

    return psi


def dpsi(inc_deriv_basis_functions, x_true, x_desired, i):
    """Taking the derivatives of the basis functions for incentives.

    :param inc_deriv_basis_functions: List of derivative basis functions for step size computation.
    :param x_true: (1 x num_players) numpy array of the true player returns.
    :param x_desired: (1 x num_players) numpy array of the desired player returns.
    :param i: Player index the target vector is being calculated for.

    return dpsi: (num_inc_deriv_basis_functions x 1) numpy array of derivatives of basis functions.
    """

    dpsi = [basis_deriv(x_true, x_desired, i) for basis_deriv in inc_deriv_basis_functions]
    dpsi = np.vstack(dpsi)

    return dpsi


def incur_loss(y_k_update, xi_k, theta_hat, num_players):
    """Incur loss from error in estimated observations using learned parameters.

    The loss is given by 1/2 * (y_i^{k+1} - xi_i^k \hat{theta}_i^{k})^2.

    :param y_k_update: (1 x num_players) numpy array of updated observations.
    :param xi_k: (num_est_basis_functions x num_players) numpy array of regression vectors.
    :param theta_hat: (num_est_basis_functions x num_players) numpy array of the estimated player parameters.
    :param num_players: Number of agents in the system.

    :return loss: (1 x num_players) numpy array of losses for players.
    """
    
    loss = np.zeros((1, num_players))
    
    for i in xrange(num_players):       
        loss[0, i] = .5*(y_k_update[0, i] - np.dot(xi_k[:, i, None].T, 
                                                   theta_hat[:, i, None]))**2
    
    return loss


def grad_loss(y_k_update, xi_k, theta_hat, i):
    """Computing gradient of the least squares loss function with respect to theta.

    The gradient is given by -xi_i^k * (y_i^{k+1} - xi_i^k \hat{theta}_i^k).
    
    :param y_k_update: (1 x num_players) numpy array of observations.
    :param xi_k: (num_est_basis functions x num_players) numpy array of regression vectors.
    :param theta_hat: (num_est_basis_functions x num_players) numpy array of the player parameters.
    :param i: Player index the loss is being calculated for.
    
    :return gradient_loss: (num_est_basis_functions x num_players) numpy array 
    of the gradient of the loss function for the player.
    """
    
    gradient_loss = np.dot(-1*xi_k[:, i, None], 
                           (y_k_update[0, i, None] - np.dot(xi_k[:, i, None].T, 
                                                            theta_hat[:, i, None])))
    return gradient_loss


def update_player_params(y_k_update, xi_k, theta_hat, num_players):
    """Update the estimated player parameters using one step of gradient descent.
    
    :param y_k_update: (1 x num_players) numpy array of updated observations.
    :param xi_k: (num_est_basis_functions x num_players) numpy array of regression vectors.
    :param theta_hat: (num_est_basis_functions x num_players) numpy array of the player parameters. 
    :param num_players: Number of agents in the system.
    
    :return theta_hat_update: (num_est_basis_functions x num_players) numpy array of new player parameters.
    """
    
    theta_hat_update = np.zeros(theta_hat.shape)

    for i in xrange(num_players):
        step = theta_step(xi_k, i)
        theta_hat_update[:, i, None] = theta_hat[:, i, None] \
                                       - step*grad_loss(y_k_update, xi_k, theta_hat, i)
    
    return theta_hat_update


def update_incentive_params(est_basis_functions, inc_basis_functions, x_true, 
                            x_desired, theta_hat_update, theta_known, nu, 
                            num_players, planner_type, gp_step):
    """Update incentive parameters by solving least squares problem for each player.

    This function solves the problem \|\zeta_i^{k+1} - \Lambda_i^k \alpha_1^{k+1}\|_2^2.

    :param est_basis_functions: List of basis functions to map return with for estimation.
    :param inc_basis_functions: List of basis functions to map return with for incentives.
    :param x_true: (1 x num_players) numpy array of the true player returns.
    :param x_desired: (1 x num_players) numpy array of the desired player returns.
    :param theta_hat_update: (num_est_basis_functions x num_players) numpy array of new estimated parameters.
    :param theta_known: (1 x num_players) numpy array of the known parameter for each player.
    :param nu: (1 x num_players) numpy array of firm knowledge parameters.
    :param num_players: Number of agents in the system.
    :param planner_type: Update method, if None update is agnostic, if 'gp' update 
    uses gradient play update.
    :param gp_step: Float step size to use for gradient play updates.
    """

    alpha = np.zeros((len(inc_basis_functions), num_players))

    for i in xrange(num_players):
        if planner_type is None:
            zeta = x_desired[:, i] - nu[:, i]*theta_known[:, i] \
                   - np.dot(phi(est_basis_functions, x_true, i).T, 
                            theta_hat_update[:, i, None]) 
            lamb = psi(inc_basis_functions, x_true, x_desired, i).T
            alpha[:, i, None] = np.linalg.lstsq(lamb, zeta)[0]
        elif planner_type == 'gp':
            zeta = x_desired[:, i] - x_true[:, i] - gp_step*(nu[:, i]*theta_known[:, i] 
                                                  + np.dot(phi(est_basis_functions, x_true, i).T, 
                                                           theta_hat_update[:, i, None])) 
            lamb = gp_step*psi(inc_basis_functions, x_true, x_desired, i).T
            check = np.linalg.lstsq(lamb, zeta)
            alpha[:, i, None] = np.linalg.lstsq(lamb, zeta)[0]

    return alpha
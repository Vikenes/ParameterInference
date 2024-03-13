import numpy as np 
import h5py
import yaml 
import os 
import emcee 
import time 


# params = np.array([
#     [13.1, 12.5, 13.8, 0.03],  # M_cut
#     [14.5, 13.5, 15.5, 0.3] ,  # M1
#     [0.8 , 0.1 , 1.5 , 0.3] ,  # sigma
#     [1.0 , 0.7 , 1.5 , 0.2] ,  # alpha
#     [0.5 , 0.0 , 1.0 , 0.25]])
# nwalkers = 10
# nparams = 5 
# p_initial = params[:, 0] + np.random.normal(size=(nwalkers, nparams)) #* params[:, 3][None, :]


# exit()
def inrange(p, params):
    return np.all((p<=params[:, 2]) & (p>=params[:, 1]))

def lnprob(p, params, param_mapping, param_tracer, Data, Ball):
    # Check if p is within the flat prior range
    if inrange(p, params):
        # read the parameters
        for key in param_mapping.keys():
            mapping_idx = param_mapping[key]
            tracer_type = param_tracer[key]
            #tracer_type = param_tracer[params[mapping_idx, -1]]
            Ball.tracers[tracer_type][key] = p[mapping_idx]
            print(key, Ball.tracers[tracer_type][key])

        # pass them to the mock dictionary
        mock_dict = Ball.run_hod(Ball.tracers, Ball.want_rsd, Nthread = 64)
        clustering = Ball.compute_xirppi(mock_dict, Ball.rpbins, Ball.pimax, Ball.pi_bin_size, Nthread = 16)
        lnP = Data.compute_likelihood(clustering)
    else:
        lnP = -np.inf
    return lnP

def main(path2config, time_likelihood):

    # load the yaml parameters
    config = yaml.load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    """ clustering_params:
    clustering_type: 'xirppi'
    bin_params:
        logmin: -0.77288
        logmax: 1.47712
        nbins: 8
    pimax: 30
    pi_bin_size: 5
    """
    data_params = config['data_params']
    ch_config_params = config['ch_config_params']
    fit_params = config['fit_params']
    """ fit_params:
    logM_cut: [0, 13.1, 12.5, 13.8, 0.03, 'LRG']
    logM1:    [1, 14.5, 13.5, 15.5, 0.3, 'LRG']
    sigma:    [2, 0.8, 0.1, 1.5, 0.3, 'LRG']
    alpha:    [3, 1.0, 0.7, 1.5, 0.2, 'LRG']
    kappa:    [4, 0.5, 0.0, 1.0, 0.25, 'LRG']
    """

    # create a new abacushod object and load the subsamples
    newBall = 0.0 # AbacusHOD(sim_params, HOD_params, clustering_params)

    # read data parameters
    newData = 0.0 # PowerData(data_params, HOD_params)

    # parameters to fit
    nparams = len(fit_params.keys()) # 5 
    param_mapping = {}
    param_tracer = {}
    params = np.zeros((nparams, 4))
    for key in fit_params.keys():
        mapping_idx = fit_params[key][0]  # 0, 1, 2, 3, 4, 5 
        tracer_type = fit_params[key][-1] # LRG
        param_mapping[key] = mapping_idx  # {'logM_cut': 0, 'logM1': 1, 'sigma': 2, 'alpha': 3, 'kappa': 4}
        param_tracer[key] = tracer_type   # {'logM_cut': 'LRG', 'logM1': 'LRG', 'sigma': 'LRG', 'alpha': 'LRG', 'kappa': 'LRG'}
        params[mapping_idx, :] = fit_params[key][1:-1] 
    """ params = np.array( 
    [[13.1, 12.5, 13.8, 0.03]  # M_cut
    [14.5, 13.5, 15.5, 0.3]   # M1
    [0.8 , 0.1 , 1.5 , 0.3]   # sigma
    [1.0 , 0.7 , 1.5 , 0.2]   # alpha
    [0.5 , 0.0 , 1.0 , 0.25]])  # kappa 
    init , min ,  max , sacling of rand-norm init param values 
    """

    # emcee parameters
    nwalkers = nparams * ch_config_params['walkersRatio']
    nsteps = ch_config_params['burninIterations'] + ch_config_params['sampleIterations']


    # fix initial conditions
    p_initial = params[:, 0] + np.random.normal(size=(nwalkers, nparams)) * params[:, 3][None, :]
    nsteps_use = nsteps
    pool_use = None # Or pool [= MPIPool()]

    # initializing sampler
    chain_file = 0.0 # SampleFileUtil(prefix_chain, carry_on=ch_config_params['rerun'])
    sampler = emcee.EnsembleSampler(nwalkers, 
                                    nparams, 
                                    lnprob, 
                                    args=(
                                        params, 
                                        param_mapping, 
                                        param_tracer, 
                                        newData, 
                                        newBall
                                        ), 
                                    pool=pool_use
                                    )
    start = time.time()
    print("Running %d samples" % nsteps_use)

    # record every iteration
    counter = 1
    for pos, prob, _ in sampler.sample(p_initial, iterations=nsteps_use):
        if True: # pool.is_master():
            print('Iteration done. Persisting.')
            chain_file.persistSamplingValues(pos, prob)

            if counter % 10:
                print(f"Finished sample {counter}")
        counter += 1

    # pool.close()
    end = time.time()
    print("Took ",(end - start)," seconds")
import numpy as np
import copy

G_SKY_PD = 0.141
G_DET_PD = 0.339
G_UNP_F = 0.287
P_SKY_PD = 0.085
P_DET_PD = 0.035
P_DET_PA = -0.38397
P_UNP_F = 0.011

def add_individual_background(settings, source):
    """
    Add background models to a fit settings, with f, q, u per detector for both photon and particle backgrounds.
    # Arguments
    * settings: the fit_settings object
    """
    obs_ids = [data.obs_id for data in settings.datas]
    obs_ids = np.unique(obs_ids)

    for obs_id in obs_ids:
        rolls = np.zeros(3)
        obs_exists = [False, False, False]

        for data in settings.datas:
            if obs_exists[data.det - 1]:
                raise Exception(f"Somehow there were two data sets for detector {data.det} and observation {obs_id}")
            obs_exists[data.det - 1] = True
            rolls[data.det-1] = data.rotation

        for det in range(1, 4):
            if not obs_exists[det - 1]: continue
            # Add the parameters
            name = f"{obs_id}bkg{det}"
            settings.add_source(copy.deepcopy(source), name, det=(det,))
            settings.set_initial_flux(name, 1)
            settings.set_initial_qu(name, (0,0))
            name = f"{obs_id}pbkg{det}"
            settings.add_particle_source(copy.deepcopy(source), name, det=(det,))
            settings.set_initial_flux(name, 1)
            settings.set_initial_qu(name, (0,0))

def add_merged_backgrounds(settings, source, initial_values=None):
    """
    Add background models to a fit settings using the background model I found.
    # Arguments
    * settings: the fit_settings object
    * initial_values: Initial values of the parameters (dictionary). Default is to use zero for all unspecified parameters.
    """
    obs_ids = [data.obs_id for data in settings.datas]
    obs_ids = np.unique(obs_ids)

    # I'm setting the PA priors to be a wider than they need be to avoid the minimizer getting stuck against a boundary.
    if initial_values is None:
        initial_values = dict()

    settings.add_param("g_sky_f", initial_values["g_sky_f"] if "g_sky_f" in initial_values else 0, (0, 1))
    settings.add_param("g_sky_pa", initial_values["g_sky_pa"] if "g_sky_pa" in initial_values else 0, (-np.pi, np.pi))
    settings.add_param("g_det_pa", initial_values["g_det_pa"] if "g_det_pa" in initial_values else 0, (-np.pi, np.pi))

    settings.add_param("p_sky_f", initial_values["p_sky_f"] if "p_sky_f" in initial_values else 0, (0, 1))
    settings.add_param("p_sky_pa", initial_values["p_sky_pa"] if "p_sky_pa" in initial_values else 0, (-np.pi, np.pi))

    for obs_id in obs_ids:
        rolls = np.zeros(3)
        obs_exists = [False, False, False]

        for data in settings.datas:
            if obs_exists[data.det - 1]:
                raise Exception(f"Somehow there were two data sets for detector {data.det} and observation {obs_id}")
            obs_exists[data.det - 1] = True
            rolls[data.det-1] = data.rotation

        for det in range(1, 4):
            if not obs_exists[det - 1]: continue
            roll = rolls[det - 1]
            settings.add_source(copy.deepcopy(source), f"{obs_id}bkg{det}", det=(det,))
            settings.set_model_fn(f"{obs_id}bkg{det}", lambda _, fit_data, params, roll=roll: (
                fit_data.param_to_value(params, "g_sky_f") * G_SKY_PD * 
                    np.cos(2*fit_data.param_to_value(params, "g_sky_pa")) + 
                (1-G_UNP_F-fit_data.param_to_value(params, "g_sky_f")) * G_DET_PD * 
                    np.cos(2*(fit_data.param_to_value(params, "g_det_pa")+roll)),
                fit_data.param_to_value(params, "g_sky_f") * G_SKY_PD * 
                    np.sin(2*fit_data.param_to_value(params, "g_sky_pa")) + 
                (1-G_UNP_F-fit_data.param_to_value(params, "g_sky_f")) * G_DET_PD * 
                    np.sin(2*(fit_data.param_to_value(params, "g_det_pa")+roll)),
            ))

            settings.add_particle_source(copy.deepcopy(source), f"{obs_id}pbkg{det}", det=(det,))
            settings.set_model_fn(f"{obs_id}pbkg{det}", lambda _, fit_data, params, roll=roll: (
                fit_data.param_to_value(params, "p_sky_f") * P_SKY_PD * 
                    np.cos(2*fit_data.param_to_value(params, "p_sky_pa")) + 
                (1-P_UNP_F-fit_data.param_to_value(params, "p_sky_f")) * P_DET_PD * 
                    np.cos(2*(P_DET_PA+roll)),
                fit_data.param_to_value(params, "p_sky_f") * P_SKY_PD * 
                    np.sin(2*fit_data.param_to_value(params, "p_sky_pa")) + 
                (1-P_UNP_F-fit_data.param_to_value(params, "p_sky_f")) * P_DET_PD * 
                    np.sin(2*(P_DET_PA+roll)),
            ))

            # Fix the polarization -- the above handles that
            settings.fix_qu(f"{obs_id}bkg{det}", (0, 0))
            settings.fix_qu(f"{obs_id}pbkg{det}", (0, 0))
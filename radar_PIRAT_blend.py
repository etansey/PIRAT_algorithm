# -*- coding: utf-8 -*-
"""
@author: etansey
"""

def increase_res(lres_vector, hres_vector):
    '''
    increase the resolution of input 'lres_vector' to match a higher res vector 'hres_vector'
    '''
    import numpy as np
    desired_res = np.ceil(len(hres_vector)/len(lres_vector))
    ltohres = np.zeros([len(hres_vector)])
    for ii in range(len(lres_vector)):
        i1 = int(ii * desired_res)
        i2 = int(i1 + desired_res)
        ltohres[i1:i2] = lres_vector[ii]
    return(ltohres)

def decrease_res(lres_vector, hres_vector):
    '''
    decrease the resolution of input 'hres_vector' to match lower res vector 'lres_vector'
    using average over timestep
    *** handy but not used as of now
    '''
    import numpy as np
    old_res = np.ceil(len(hres_vector)/len(lres_vector))
    htolres = np.zeros([len(lres_vector)])
    for ii in range(len(lres_vector)):
        i1 = int(ii * old_res)
        i2 = int(i1 + old_res)
        hres_mean = np.mean(hres_vector[i1:i2])
        htolres[ii] = hres_mean
    return(htolres)

# >>> function to find nearest value in an array
def find_nearest(array, value):
    import numpy as np
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def blend(type_tipr, R, R_scaled, t_missing, r_eff_tipr, r_eff_tipr_scaled, divisorx, t_tipr, TB, t_aws, dq_radar, R_radar, r_eff_radar, t_radar):
    '''
    blending radar & output from TIPR subroutines for comparison to tipping bucket.

    dimensions: t_tipr (x-minute res TIPR data) ; t_aws (1-minute res met data) ; t_radar (12-sec radar data)

    INPUT
    1) type_tipr == phase of precip, output from TIPR
    dimension: t_tipr ; unitless
    2) R == unscaled spectrum precip rate using the believed phase of precip, output from TIPR
    dimension: t_tipr ; unit: mm/hour
    3) R_scaled == scaled spectrum precip rate using the believed phase of precip, output from TIPR
    dimension: t_tipr ; unit: mm/hour
    4) r_eff_tipr == unscaled spectrum mean effective radius from an x-minute spectrum, output from TIPR
    dimension: t_tipr ; unit: microns
    5) r_eff_tipr_scaled == scaled spectrum mean effective radius from an x-minute spectrum, output from TIPR
    dimension: t_tipr ; unit: microns
    6) divisorx == designated TIPR time resolution; ex: for 5-min res, divisorx = 5.
    dimensionless, unitless
    7) TB == tipping bucket (not used explicitly in algorithm; only shows up in 'act_trckr')
    dimension: t_aws ; unit: mm
    8) t_aws == AWS metorology data time array (not used explicitly in algorithm; only shows up in 'act_trckr')
    dimension: t_aws ; unit: hour
    9) dq_radar == radar data quality flag (bit-packed)
    dimension: t_radar ; unitless
    10) R_radar == radar precip rate
    dimension: t_radar ; unit: mm/hr
    11) r_eff_radar == radar effective radius
    dimension: t_radar ; unit: microns
    12) t_missing == 1440 minutes (1-minute resolution over 1 day) accounting for parsivel's missing places

    OUTPUT
    1) r_eff_blend == efective radius designated by blending algorithm
    dimension: t_blend ; unit: [microns]
    2) R_blend == rate designated by blending alg
    dim: t_blend ; unit: [mm/hr]
    3) t_blend == time array to pair with blend output variables
    4) act_trckr == list of potentially useful variables for tracking what blend algorithm did
        - order of variables:
            1. 'act' -- corresponds to which conditional the algorithm used for this timestep
            2. time in hours of this timestep
            3. TIPR precip type at this timestep, if any (0 = 'no weather')
            4. tipping bucket accumulation at this timestep
            5. cumulative blend accumulation thus far
            6. radar mean rate under consideration at this timestep
            7. TIPR rate underconsideration at this timestep
    5) R_radar_lres == array of radar rates used in blend algorithm; lower resolution using mean of 12-second rates.
    dimension: t_blend ; unit: [mm/hour]

    '''
    import numpy as np
    # >>> increase time TIPR from x-minute resolution to 1-minute
    R_hres, R_scaled_hres = increase_res(R, t_missing), increase_res(R_scaled, t_missing)
    hrestTIPR = increase_res(t_tipr, t_missing)
    hrestypeTIPR, hresr_eff_TIPR, hresr_eff_TIPR_scaled = increase_res(type_tipr, R_hres), increase_res(r_eff_tipr, R_hres), increase_res(r_eff_tipr_scaled, R_hres)

    # >>> blend outputs:
    R_blend, t_blend, r_eff_blend = [np.zeros([len(hrestTIPR)]) for i in range(3)]
    act_trckr = ["" for x in range(len(hrestTIPR))] # >>> useful for tracking what action the blend algorithm took and why

    # >>> R_hres - length radar rates using means:
    R_radar_lres = np.zeros([len(R_scaled_hres)])

    for ii in np.arange(0, len(hrestTIPR), divisorx):
        TBt = find_nearest(t_aws, hrestTIPR[ii])
        TBidx = int(np.where(t_aws==TBt)[0])
        accum_blend_tmp = np.cumsum(R_blend * (1/60)) # >>> current accumulation of R_blend so far
        jj = ii+divisorx # >>> increase time of TIPR from x-minute resolution to 1-minute
        if jj >= len(hrestTIPR):
            jj = len(hrestTIPR)-1
        t1, t2 = hrestTIPR[ii], hrestTIPR[jj]
        t_blend[ii:jj] = t1 # >>> time array for _blend variables
        # >>> SYNCHRONIZE RADAR VARIABLES
        tr1, tr2 = find_nearest(t_radar[:], t1), find_nearest(t_radar[:], t2) # >>> align radar times
        ridx1, ridx2 = int(np.where(t_radar==tr1)[0]), int(np.where(t_radar==tr2)[0])
        dq_radar_tmp = dq_radar[ridx1:ridx2]
        # >>> Rr_tmp, rr_tmp are arrays of radar rate & radius for across time t1:t2
        Rr_tmp = np.copy(R_radar[int(np.where(t_radar==tr1)[0]):int(np.where(t_radar==tr2)[0])])
        rr_tmp = np.copy(r_eff_radar[int(np.where(t_radar==tr1)[0]):int(np.where(t_radar==tr2)[0])])
        Rr_tmp[np.isnan(Rr_tmp)], rr_tmp[np.isnan(rr_tmp)] = 0., 0.
        if len(Rr_tmp) == 0.:
            Rr_mean, r_effr_mean = 0., 0.
        elif len(Rr_tmp) > 0.:
            Rr_mean, r_effr_mean = np.mean(Rr_tmp), np.mean(rr_tmp) # >>> mean radar rate, effective radius
        # >>> save mean rate to lower-resolution radar rates array
        R_radar_lres[ii] = Rr_mean
        if np.isnan(Rr_mean):
            Rr_mean = 0.
        # >>> which TIPR rate to use:
        if (abs(Rr_mean-R_scaled_hres[ii]) <= abs(Rr_mean-R_hres[ii])):
            Rtipr1, typetmp, r_efftipr = R_scaled_hres[ii], hrestypeTIPR[ii], hresr_eff_TIPR_scaled[ii] # >>> TIPR rate, type, eff r
#            print("%.2f" % t1,' Rbest2 chosen, diff = %.4f' % (abs(Rr_mean-R_scaled_hres[ii])))
        elif (abs(Rr_mean-R_scaled_hres[ii]) > abs(Rr_mean-R_hres[ii])):
            Rtipr1, typetmp, r_efftipr = R_hres[ii], hrestypeTIPR[ii], hresr_eff_TIPR[ii] # >>> TIPR rate, type, eff r
#            print("%.2f" % t1,' Rbest chosen, diff = %.4f' % (abs(Rr_mean-R_scaled_hres[ii])))
            if ((len(type_tipr[type_tipr==5.])!=0) and hrestypeTIPR[ii]!=5. and len(type_tipr[(type_tipr!=5.) & (type_tipr!=0)])/len(type_tipr[type_tipr==5.])>=0.25) \
            or ((len(type_tipr[type_tipr==5.])!=0) and len(type_tipr[(type_tipr!=5.) & (type_tipr!=0) & (type_tipr!=1.)])<=1. and len(type_tipr[(type_tipr!=5.) & (type_tipr!=0)])/len(type_tipr[type_tipr==5.])>=0.25):
                Rtipr1, typetmp, r_efftipr = R_scaled_hres[ii], hrestypeTIPR[ii], hresr_eff_TIPR_scaled[ii]
#               print("%.2f" % t1,' actually Rbest2 chosen, diff = %.4f' % (abs(Rr_mean-R_scaled_hres[ii])))
        else:
            Rtipr1, typetmp, r_efftipr = R_scaled_hres[ii], hrestypeTIPR[ii], hresr_eff_TIPR_scaled[ii]

        if (np.abs(tr1 - t1) >= 0.05): # >>> no radar data at this timestamp
            R_blend[ii:jj], r_eff_blend[ii:jj] = Rtipr1, r_efftipr
            act_trckr[ii] = 1., t1, int(typetmp), TB[TBidx], float(accum_blend_tmp[ii-1]+R_blend[ii]*(1/60)), Rr_mean, Rtipr1
            continue
        else:
            if (len(dq_radar_tmp[dq_radar_tmp!=0])  >= 0.5 * len(dq_radar_tmp)) and \
            (Rr_mean > np.median(R_radar[(dq_radar==0.) & (R_radar!=0.) & (~np.isnan(R_radar))])): # >>> DQ flag of radar nonzero! assign Rr_mean an arbitrary value of 0.001 mm/hr
                Rr_mean = np.median(R_radar[(dq_radar==0.) & (R_radar!=0.) & (~np.isnan(R_radar))])
                if np.median(R_radar[(dq_radar==0.) & (R_radar!=0.) & (~np.isnan(R_radar))]) > 0.5:
                    Rr_mean = 0.001

            if (Rtipr1 > 1E-10) and ((Rr_mean < 1E-10) or (np.isnan(Rr_mean))): # >>> TIPR rate alone
                R_blend[ii:jj], r_eff_blend[ii:jj] = Rtipr1, r_efftipr
                act_trckr[ii] = 3., t1, int(typetmp), TB[TBidx], float(accum_blend_tmp[ii-1]+R_blend[ii]*(1/60)), Rr_mean, Rtipr1

            if (Rtipr1 < 1E-10) and (1E-10 < Rr_mean < 5.):# >>> no TIPR rate
                R_blend[ii:jj], r_eff_blend[ii:jj] = Rr_mean, r_effr_mean
                act_trckr[ii] = 4., t1, int(typetmp), TB[TBidx], float(accum_blend_tmp[ii-1]+R_blend[ii]*(1/60)), Rr_mean, Rtipr1

            if (typetmp == 5.) and (Rtipr1 > 7.): # >>> type is small, rate from TIPR unreasonably high
                R_blend[ii:jj], r_eff_blend[ii:jj] = Rr_mean + 0.5*Rtipr1, r_effr_mean
                act_trckr[ii] = 5., t1, int(typetmp), TB[TBidx], float(accum_blend_tmp[ii-1]+R_blend[ii]*(1/60)), Rr_mean, Rtipr1

            if (typetmp == 5.) and (Rtipr1 <= 7.): # >>> type is small, rate from TIPR not too high
                R_blend[ii:jj], r_eff_blend[ii:jj] = 0.5*Rr_mean + Rtipr1, r_effr_mean
                act_trckr[ii] = 5.4, t1, int(typetmp), TB[TBidx], float(accum_blend_tmp[ii-1]+R_blend[ii]*(1/60)), Rr_mean, Rtipr1
                if (Rr_mean == np.median(R_radar[(dq_radar==0.) & (R_radar!=0.)])):
                    R_blend[ii:jj], r_eff_blend[ii:jj] = Rr_mean + Rtipr1, r_effr_mean
                    act_trckr[ii] = 5.5, t1, int(typetmp), TB[TBidx], float(accum_blend_tmp[ii-1]+R_blend[ii]*(1/60)), Rr_mean, Rtipr1


            if (typetmp == 0.) and (1E-10 < Rr_mean < 5.): # >>> type='no weather', but there is a rate from radar
                R_blend[ii:jj], r_eff_blend[ii:jj] = Rr_mean, r_effr_mean
                act_trckr[ii] = 6., t1, int(typetmp), TB[TBidx], float(accum_blend_tmp[ii-1]+R_blend[ii]*(1/60)), Rr_mean, Rtipr1

            if (1 <= typetmp < 5.) and (r_efftipr >= 400.): # >>> all types besides small, TIPR radius too high for radar to be useful
                R_blend[ii:jj], r_eff_blend[ii:jj] = Rtipr1, r_efftipr
                act_trckr[ii] = 7., t1, int(typetmp), TB[TBidx], float(accum_blend_tmp[ii-1]+R_blend[ii]*(1/60)), Rr_mean, Rtipr1

            if (1 <= typetmp < 5.) and (Rr_mean >= 5.): # >>> all types besides small, radar rate too high to be trusted
                R_blend[ii:jj], r_eff_blend[ii:jj] = Rtipr1, r_efftipr
                act_trckr[ii] = 8., t1, int(typetmp), TB[TBidx], float(accum_blend_tmp[ii-1]+R_blend[ii]*(1/60)), Rr_mean, Rtipr1

            if ((1 <= typetmp < 5.) and (Rr_mean < 5.)) or ((1 <= typetmp < 5.) and (r_effr_mean < 200.)):
                # >>> all types besides small, radar rate/radius trustworthy
                R_blend[ii:jj], r_eff_blend[ii:jj] = 0.8 * Rtipr1 + 0.5*Rr_mean, r_efftipr
                act_trckr[ii] = 8.3, t1, int(typetmp), TB[TBidx], float(accum_blend_tmp[ii-1]+R_blend[ii]*(1/60)), Rr_mean, Rtipr1
                if Rr_mean == np.median(R_radar[(dq_radar==0.) & (R_radar!=0.)]): # >>> potentially delete -- 17.8.19
                    R_blend[ii:jj], r_eff_blend[ii:jj] = Rtipr1 + Rr_mean, r_efftipr
                    act_trckr[ii] = 8.4, t1, int(typetmp), TB[TBidx], float(accum_blend_tmp[ii-1]+R_blend[ii]*(1/60)), Rr_mean, Rtipr1
                if (Rtipr1 >= 15.) and (Rtipr1 >= np.median(R[(type_tipr!=0.) & (type_tipr!=5.)]) \
                + 3*np.std(R[(type_tipr!=0.) & (type_tipr!=5.)])): # >>> potentially delete -- 17.8.19
                    R_blend[ii:jj] = 0.6 * Rtipr1 + 0.4*Rr_mean
                    act_trckr[ii] = 8.5, t1, int(typetmp), TB[TBidx], float(accum_blend_tmp[ii-1]+R_blend[ii]*(1/60)), Rr_mean, Rtipr1

            if (Rtipr1!= 0.) and (Rtipr1>Rr_mean) and (Rtipr1 - Rr_mean <= 0.1):
                R_blend[ii:jj], r_eff_blend[ii:jj] = Rtipr1, r_efftipr
                act_trckr[ii] = 9., t1, int(typetmp), TB[TBidx], float(accum_blend_tmp[ii-1]+R_blend[ii]*(1/60)), Rr_mean, Rtipr1

            if (R_blend[ii] > 1E-10) and (r_eff_blend[ii] < 1E-10):  # >>> make sure r_eff_blend has a value, even if from radar
                r_eff_blend[ii:jj] = r_effr_mean

    t_blend[len(t_blend)-1] = t_blend[len(t_blend)-2] # >>> ensure last val of t_blend is ~24 hours

    # >>> if parsivel didn't measure anything or it measured very little and all 'small', rely on radar rates.
    if (abs(len(type_tipr[type_tipr==0.]) - len(type_tipr)) <= 1.):
        R_blend, t_blend = np.copy(R_radar_lres), np.copy(hrestTIPR)
    if (abs(len(type_tipr[type_tipr==0.]) - len(type_tipr)) <= 3.) and \
    (len(type_tipr[type_tipr!=0.]) == len(type_tipr[type_tipr==5.])):
        R_blend, t_blend = np.copy(R_radar_lres), np.copy(hrestTIPR)
    # >>> a day with only large-particle precip (no 'small') should rely on TIPR exclusively
    if (len(type_tipr[type_tipr==5.])==0.) and (len(type_tipr[type_tipr!=0.]) >= 4.):
        R_blend, t_blend = np.copy(R_scaled_hres), np.copy(hrestTIPR)

    return(r_eff_blend, R_blend, t_blend, act_trckr, R_radar_lres)

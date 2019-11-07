
# coding: utf-8

# In[13]:


"""
Created on Fri Sep 28 17:15:51 2018
@author: etansey

 - 5 min spectra composited. If a 5 min spec has < 100 counts, it's considered no weather.

 - Values m_t (t = type) for rain; ice pellets; snow; wet snow; small (phase unknown, majority particles <1mm)
where m_t is a metric to indicate closeness of a count-filled bin's velocity midpoint
to the myfit curves.

 - Weather code will be determined by closest curve (largest m_t):
 RAIN = 1
 ICE PELLETS = 2
 SNOW = 3
 WET SNOW = 4
 SMALL = 5

 - Subsequently, precipitation rate, DSDs calculated

 - Note: precip rates, accumulation returned for all phases, regardless of weather code. Rate array
 that takes weather code into account is also returned.

FILE LAST UPDATED: 20190804
# >>> History:
# >>>     20190707 global variable "ix" from subroutine specshift; changed 32s to n_v, n_D
# >>>     20190708 added commenting to beginning of subroutines to describe dimensions, units of variables input, returned
# >>>     20190711 separate subroutine for x minute averaging (previously 5 min, now defined as input variable)
# >>>     20190715 revised badvals and entire 'change_resolution' function
# >>>     20190716 revised m_1mm sequence so that metrics for all other types are still output, even if phase is "small"
                -  if metrics m_r, m_ip, m_s and m_ws are zero, it means no particles greater than 1mm were detected.
            - Changed output to include effective radius for each x-minute spectrum.
# >>>     20190718 revised m_1mm sequence so that it's a ratio of low-diameter particles' volume/big-diameter volumes
                - threshold: if more than 1% of x-minute spectrum's volume (assumed liquid) is small, weather code=5 ('small')
# >>>     20190722 introduce data quality flagging; import Python module "bitsets" to my subroutines and flag as follows:
                - bit 1 = 'missing_all_spectra' means all spectra (x minute resolution) are missing in a
                timestep due to instrument issues
                - bit 2 = 'missing_n_of_x_spectra' means out of x minutes, a number of measurements are missing;
                histogram of zeros was added to pad missing time
                - bit 3 = 'renormalized' before spectral calculations (DSD, effective diameter, rate),
                counts from noise regions are added to median bin
                    --only done if wind or margin fallers noise is > 50% of the counts.
# >>>     20190724 changed to V1_5
                    - set threshold for when renormalization should occur
                        - 30% or more of counts should be in non-noise regions
                    - commented out comparison to Parsivel vendor's precip rate (towards end of rates_calculation)
                    - found an error. renormalization was using noise region counts along velocity axis, not diameter.
# >>>     20190801 metrics comparison for rain and wet snow
                    - to avoid overaccumulation due to misclassification of precip as completely liquid:
                        Metrics wet snow (m_ws) and rain (m_r) compared. If (m_r - m_ws) <= 0.15, the 5min spectrum
                        is assumed to contain some wet snow. High density wet snow correction applied.
# >>>     20190804 only 25% of counts used in renormalization.
# >>>     20190805 changed confines of wind mask/ expanded region considered 'snow'.
                    - changed specshift requirements (above stated, margin fallers, and potential ice/snow region)
# >>>     20190920 subroutine 'scale_spectra' added
                    - increase counts in spectra by multiplying with constant
                    - different scaling factors for different laser amplitude ranges, phases
# >>>     20191102 fixed areas where "divide by zero" and "mean/median of empty array" warnings occur
"""

# In[3]:
# >>> small functions used in subroutines:
# >>> make # of minutes divisible by 1440
def round_down(num, divisor):
        return num - (num%divisor)

# >>> increase the resolution of input 'lres_vector' to match a higher res vector 'hres_vector'
def increase_res(lres_vector, hres_vector):
    import numpy as np
    desired_res = np.ceil(len(hres_vector)/len(lres_vector))
    ltohres = np.zeros([len(hres_vector)])
    for ii in range(len(lres_vector)):
        i1 = int(ii * desired_res)
        i2 = int(i1 + desired_res)
        ltohres[i1:i2] = lres_vector[ii]
    return(ltohres)

# >>> a function to take x-minute avgs of full resolution variables
def var_xmin(input_array, divisorx, t_input_length):
    import numpy as np
    dummyy = np.zeros([t_input_length])
    for j in range(t_input_length):
        j1 = j*divisorx
        j2 = j1 + divisorx
        dummyy[j] = np.nansum(input_array[j1:j2])/divisorx
    return dummyy

# >>> function to find nearest bin, should a spectral shift be necessary
def find_nearest(array, value):
    import numpy as np
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
# >>> running mean to be used on laser amplitude
def running_mean(x, N):
    import numpy as np
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return( (cumsum[N:] - cumsum[:-N]) / float(N) )
# In[]
# SUBROUTINE #0
def change_resolution(time, precip_rate, spectrum_array, amp, raw_fall_velocity, particle_size, divisorx):
    '''
    # decreasing time resolution of 1-minute Parsivel variables.

    VARIABLES INPUT:
# time (from Parsivel vendor)
        ## dimension: time ; unit: [seconds]
# spectrum_array (from Parsivel vendor)
        ## dimension: {time, raw_fall_velocity, particle_size} ; unit: [counts]
# divisorx == what time resolution do you want?
        dimension: integer input
# amp == laserband amplitude (from Parsivel vendor)
        ## dimension: time ; unit: [arbitrary counts]

    VARIABLES RETURNED:
# rawspecxmin = raw spectrum with x-min resoltion size-velocity histograms (no accounting for wind, margin fallers, noise)
        ## dimensions: {n_tx, n_v, n_D}; units: [counts]
# txmin = shifted, noise-corrected spectra (accounts for wind, margin fallers)
        ## dimensions: n_tx, or time/divisorx; units: [seconds]
# tmissing = copy of Parsivel's 'time' array with NANs in places where the instrument cut out
        ## dimensions: 1440 ; units: [seconds]
# prate_missing = copy of Parsivel's 'precip_rate' array with zeros in places where the instrument cut out
        ## dimensions: 1440 ; units: [mm/hour]
# DQflag = data quality flag; see preamble for description.
        ## dimensions: n_tx ; unitless
# amp_missing = copy of Parsivel's 'laserband_amplitude' array with NANs in places where the instrument cut out
        ## dimensions: 1440 ; units: [arb. counts]

'''
    import numpy as np
    from bitsets import bitset
#    from Pdata_subroutined_V1_2 import round_down, find_nearest
    # >>> get necessary stuff from mcqpars2S1.b1 .nc file
    n_t = len(time)  # >>> total # of 1-min intervals in present dataset
    n_v = len(raw_fall_velocity)   # >>> 32 raw_fall_velocity bins
    n_D = len(particle_size)   # >>> 32 particle_size bins
    # >>> generic day array; units: [seconds];  timestep: 60 seconds; length: 1440
    tday = np.arange(0, 86400, 60)
    # >>> Matlab requires passing in array's not matricies ... so need to unpack array
    raw_spectrum = np.reshape(spectrum_array, (n_t, n_v, n_D), order='F')
    dummyy = np.array(raw_spectrum, dtype=np.float)
    dummy = dummyy.clip(min=0)
    # >>> dimension of new resolution set of spectra will be 1440/divisorx
    n_tx = int(len(tday) / divisorx)
    # >>> data quality flag setup:
    DQ_FLAGS = ('missing_all_spectra', 'missing_n_of_x_spectra', 'renormalized', 'laser_calibration')
    DQ_flags = bitset('dq_flags', DQ_FLAGS)
    DQ_array, mssd_specs, badvals = np.zeros([n_tx]), np.zeros([len(tday)]), []
    if len(time) < len(tday):
        for ii in range(len(tday)):
            if tday[ii] > time[-1]:
                badvals.append(ii)
                continue
            elif tday[ii] < time[0]:
                badvals.append(ii)
                continue
#            elif (ii<len(time)-1) and (time[ii+1] - time[ii] != 60.):
#                badvals.append(ii)
    if len(badvals) > 0:
        timefix, dummy_empties, prate_missing, amp_missing = np.copy(tday), np.zeros([len(tday), n_v, n_D]), np.zeros([len(tday)]), np.zeros([len(tday)])
        timefix[badvals] = -999
        for ii in range(len(timefix)):
            if timefix[ii] == -999:
                mssd_specs[ii] = 1.
            if timefix[ii] != -999:
                dummy_empties[ii,:,:] = np.copy(dummy[time == timefix[ii]])
                prate_missing[ii] = np.copy(precip_rate[time == timefix[ii]])
                amp_missing[ii] = np.copy(amp[time == timefix[ii]])

    if len(badvals) == 0:
        timefix, dummy_empties, prate_missing, amp_missing = np.copy(time), np.copy(dummy), np.copy(precip_rate), np.copy(amp)
    # >>> create divisorx-minute resolution spectrum:
    rawspecxmin = np.zeros([n_tx, n_v, n_D])
    for ii in range(n_tx):
        i1 = ii*divisorx
        i2 = i1 + divisorx
        for kk in range(n_D):
            for jj in range(n_v):
                rawspecxmin[ii, jj, kk] = np.ma.sum(dummy_empties[i1:i2, jj, kk])
        # >>> check if Parsivel cut out during timesteps i1:i2 during measurement. If so, DQ flag.
        ck = np.copy(mssd_specs[i1:i2])
        ck2 = len(ck[ck==1.])
        if ck2 == 0:
            continue
        else:
            if ck2 == 5.:
                tmp = DQ_flags(['missing_all_spectra'])
            if 0. < ck2 < 5.:
                tmp = DQ_flags(['missing_n_of_x_spectra'])
            int_tmp = int(tmp)
            DQ_array[ii] = int_tmp
        del(ck, ck2)

    # >>> put NANs anywhere that the Parsivel's measurement time array missed
    tmissing = np.copy(timefix)
    for ii in range(len(tmissing)):
        if tmissing[ii] == -999.:
            amp_missing[ii] = np.nan
    # >>> get x-minute time array; dimension: n_tx ; unit: [seconds]
    txmin = var_xmin(tday, divisorx, n_tx)

    return(rawspecxmin, tmissing, prate_missing, amp_missing, txmin, DQ_array)

# In[]
# SUBROUTINE #1
def specshift(txmin, rawspecxmin, raw_fall_velocity, particle_size):
    '''
    # spectral corrections algorithm.

    LOCAL VARIABLES RE-ASSIGNED THROUGHOUT SUBROUTINES:
# Local variables for dimensions:
    ## n_t = daily total 1 minute spectra
    ## n_tx = daily total x min spectra
    ## n_v = 32 velocity bins
    ## n_D = 32 diameter bins
# X, Y = diameter, velocity axes ; D = midpoints of diameter bins ; Ymid = velocity midpoints
# precip_rate = parsivel vendor's precip rate
        ## dimensions: time; units: [mm/hour]
# v_GK, v_ip, v_mixd, v_snow = rain (empirical, Gunn & Kinzer 1958), ice pellets, mixed, and snow fall speeds as a function of diameter
        ## units: [m/s]
# wind, margin_fallers = masks to be applied over size-velocity spectra to tally counts in noise regions
# mf_ratios, w_ratios = ratios of counts in margin fallers (mf) bins and wind (w) bins / total counts in spec

    VARIABLES INPUT:
# time, raw_fall_velocity, particle_size (from Parsivel vendor)
# txmin
        ## dimension: time/x ; units: [seconds]
# rawspecxmin
        ## dimension: {txmin, raw_fall_velocity, particle_size} ; unit: [counts]

    VARIABLES RETURNED:
# rawspecxmin = raw spectrum with x-min resoltion size-velocity histograms (no accounting for wind, margin fallers, noise)
        ## dimensions: {n_tx, n_v, n_D}; units: [counts]
# shiftspec = shifted, noise-corrected spectra (accounts for wind, margin fallers)
        ## dimensions: {n_tx, n_v, n_D}; units: [counts]

    '''
    import numpy as np
    # >>> get necessary stuff from mcqpars2S1.b1 .nc file
    n_tx = len(txmin)  # >>> total # of 1-min intervals in present dataset
    n_v = len(raw_fall_velocity)   # >>> 32 raw_fall_velocity bins
    n_D = len(particle_size)   # >>> 32 particle_size bins
    # >>> Matlab requires passing in array's not matricies ... so need to unpack array
    #____________________________________________________________________________
    # >>> make x,y axes for all plots
    X, Y = np.meshgrid(particle_size, raw_fall_velocity)
    # >>> midpoints of vel, size bins
    x = X[0,:]
    D = np.array((x[1:] + x[:-1]) / 2).tolist()
    D.insert(0, x[0]/2)
    D = np.array(D[:])
    y = Y[:,0]
    Ymid = np.array((y[1:] + y[:-1]) / 2).tolist()
    Ymid.insert(0, y[0]/2)
    Ymid = np.array(Ymid[:])
    # >>> my fits: phase-specific expressions for fall velocities
    a_s, b_s = 1.29129876915, 0.352624015539
    v_snow = a_s * D ** b_s
    v_GK = 9.65 - 10.3 * np.e**(-0.6 * D)  # >>> rain fall speed equation used thruout
    a_ip, b_ip = 2.47563950652, 0.249683670041
    v_ip = a_ip * D ** b_ip
    # >>> spectral shifting as function of mf, w ratios
    # >>> coefficients from polyfit of rain errors
    p0, p1, p2, p3 = -0.15700349,  1.25494849, -1.96797664,  2.90517874
    #____________________________________________________________________________
    # >>> MASKING WINDY AND MARGIN FALLERS REGIONS
    rmask = np.zeros([n_v,n_D])   # >>> rain mask; only diameters < bin#25
    for i in range(n_D - 7):
        v_temp = float(v_GK[i])
        for j in range(n_v):
            if (float(Ymid[j]) <= v_temp - 0.2*v_temp) or (float(Ymid[j]) >= v_temp + 0.4*v_temp):
                continue
            else:
                rmask[j,i] = 1
    sIDmask = np.zeros([n_v,n_D])   # >>> snow mask
    for i in range(11,n_D - 7):
        v_temp = float(v_ip[i])
        for j in range(n_v):
            if (float(Ymid[j]) <= v_temp - 0.7*v_temp) or (float(Ymid[j]) >= v_temp + 0.05*v_temp):
                continue
            else:
                sIDmask[j,i] = 4
    wind = np.zeros([n_v,n_D])   # >>> wind-induced counts; diameter bins 6+
    for i in range(9,n_D):
        v_temp = float(v_snow[i])
        for j in range(n_v):
            if float(Ymid[j]) <= v_temp - 0.7*v_temp:
                wind[j,i] = 2
    margin_fallers = np.zeros([n_v,n_D])
    for i in range(16):
        for j in range(4,n_v):
            if (float(rmask[j,i]) == 0) and (float(Ymid[j]) >= float(v_GK[i])):
                margin_fallers[j,i] = 3
    #____________________________________________________________________________
    # >>> RATIOS OF COUNTS IN MARGIN FALLERS, WINDY REGIONS
    mf_ratios = np.zeros(n_tx)
    w_ratios = np.zeros(n_tx)
    for k in range(n_tx):
        spec = np.copy(rawspecxmin[k,:,:])
        if np.sum(spec) == 0:
            continue
        else:
            mf_dummy = np.zeros([n_v,n_D])
            w_dummy = np.zeros([n_v,n_D])
            for i in range(n_D):
                for j in range(n_v):
                    if margin_fallers[j,i] == 3:
                        mf_dummy[j,i] = spec[j,i]
                    elif wind[j,i] == 2:
                        w_dummy[j,i] = spec[j,i]
            mf_ratios[k] = float(np.sum(mf_dummy)/np.sum(spec))
            w_ratios[k] = float(np.sum(w_dummy)/np.sum(spec))
    #____________________________________________________________________________
    # >>> spectral shifting as function of mf, w ratios
    shiftspec = np.zeros([n_tx,n_v,n_D])
    shift_track = np.zeros(n_tx)   # >>> ones where a shift was applied
    for timestep in range(n_tx):
        # >>> full spectrum including D<=1mm, without noise
            TEMPspec = np.copy(rawspecxmin[timestep,:,:])
            for p in range(n_D):
                for pp in range(n_v):
                    if wind[pp,p] == 2:
                        TEMPspec[pp,p] = 0
                    if (margin_fallers[pp,p] == 3):
                        TEMPspec[pp,p] = 0
            if (np.sum(TEMPspec[8:n_v,20:n_D]) > 3.): # >>> big particles falling at reasonable speeds -- frozen; no shift.
                shiftspec[timestep,:,:] = np.copy(TEMPspec[:,:])
                shift_track[timestep] = 1.
                continue
            elif (np.sum(TEMPspec[sIDmask==4.]) >=20.): # >>> many counts in sIDmash region -- frozen; no shift.
                shiftspec[timestep,:,:] = np.copy(TEMPspec[:,:])
                shift_track[timestep] = 1.5
                continue
            elif ((mf_ratios[timestep] >= 0.001) and (np.sum(TEMPspec[sIDmask==4.])<20.)) \
                  or (np.sum(TEMPspec[wind==2]) >= 3.):
                # >>> very high margin fallers ratio -- likely liquid. shift.; wind region contamination -- shift.
                shift_track[timestep] = 2.
                for i in range(n_D):   # >>> only up to reasonable raindrop size, through (5.125 mm)
                    if (0 <= i <= 6) or (21 <= i <= n_D):
                        shiftspec[timestep,:,i] = np.copy(TEMPspec[:,i])
                        continue
                    else:
                        vGK_temp = float(v_GK[i])
                        for j in range(n_v - 5):
                            N = TEMPspec[j,i]
                            if N < 0.001:
                                continue
                            elif float(Ymid[j]) >= vGK_temp:
                                shiftspec[timestep,j,i] = np.copy(TEMPspec[j,i])
                                continue
                            else:
                                pfit = float(p3 + p2*float(D[i]) + p1*float(D[i])**2 + p0*float(D[i])**3)
                                du = pfit * (1/2) + float(Ymid[j])
                                if 6 <= i <= 13:
                                    du = pfit + float(Ymid[j])
                                new_bin = find_nearest(Ymid[:],du)
                                for k in range(n_v):
                                    Y1, Y2 = float(Y[k-1,0]), float(Y[k,0])
                                    if (Y1 < new_bin < Y2) and (new_bin <= vGK_temp + 1.):
                                        shiftspec[timestep,k,i] = np.copy(TEMPspec[k,i]) + N
                                    elif (Y1 < new_bin < Y2) and (new_bin > vGK_temp + 1.):
                                        shiftspec[timestep,k-2,i] = np.copy(TEMPspec[k-2,i]) + N  # >>> move it down a little if shifted too high
                continue
            else:
                shiftspec[timestep,:,:] = np.copy(TEMPspec[:,:])
                shift_track[timestep] = 3.
                continue

    # >>> lastly, set noise regions to zero before metrics calculated.
    for p in range(n_D):
        for pp in range(n_v):
            if wind[pp,p] == 2:
                shiftspec[:,pp,p] = 0
            if (margin_fallers[pp,p] == 3):
                shiftspec[:,pp,p] = 0

    return(shiftspec, w_ratios, mf_ratios, shift_track)


# In[40]:
# SUBROUTINE #2
def metrics(txmin, particle_size, raw_fall_velocity, shiftspec, DQ_array, rawspecxmin, w_ratios, mf_ratios):
    '''
    # calulate metrics; generate weather codes

    VARIABLES INPUT:
# time, shiftspec, raw_fall_velocity, particle_size

    VARIABLES RETURNED:
# m_r m_ip, m_s, m_ws = metrics calculated using statistical method, proximity to phase-specific fall speed equations
# m_1mm_const = zeros and ones ; value of 1 means 95% of COUNTS are in D bins smaller than 1mm
# m_1mm_vols = ratio of VOLUMES ; volume small diameters / volume big diameters
# my_wcs = weather codes found by either the largest metric, or if most counts and >1% of liquid volume is in small D bins
# cws_designator = designates which density correction should be used in 'rates_calculation' subroutine:
    # Compare abs(m_ice - m_wetsnow) and abs(m_rain - m_wetsnow). If wet snow metric is more similar to ice pellets, use
    # a less dense coefficient (called 'cws') in rates calculation. And vice versa.

        ## dimension of all returned variables: n_tx (number of x-minute spectra)
        ## all are unitless
    '''

    import numpy as np
    from bitsets import bitset
    # >>> GET LOCAL VARIABLES AGAIN
    n_tx = len(txmin)  # >>> total # of 1-min intervals in present dataset
    n_v = len(raw_fall_velocity)   # >>> 32 raw_fall_velocity bins
    n_D = len(particle_size)   # >>> 32 particle_size bins
    X, Y = np.meshgrid(particle_size, raw_fall_velocity)
    # >>> midpoints of vel bins (Ymid) and size bins (D)
    x, y = X[0,:], Y[:,0]
    D = np.array((x[1:] + x[:-1]) / 2).tolist()
    D.insert(0, x[0]/2)
    D = np.array(D[:])
    Ymid = np.array((y[1:] + y[:-1]) / 2).tolist()
    Ymid.insert(0, y[0]/2)
    Ymid = np.array(Ymid[:])
    # >>> my fits: phase-specific expressions for fall velocities
    a_ip, a_s, a_rip = 2.47563950652,  1.29129876915, -13.9193270007
    b_ip, b_s, b_rip = 0.249683670041, 0.352624015539, -0.34355822607
    c_rip = -1.78307830508
    # >>> fall velocity equations
    v_snow = a_s * D ** b_s
    v_ip = a_ip * D ** b_ip
    v_mixd = c_rip - a_rip * np.exp(-D ** b_rip)
    v_GK = 9.65 - 10.3 * np.e**(-0.6 * D)  # >>> rain fall speed equation used thruout
       #____________________________________________________________________________
    # >>> MASKING WINDY AND MARGIN FALLERS REGIONS
    rmask = np.zeros([n_v,n_D])   # >>> rain mask
    for i in range(n_D - 7):
        v_temp = float(v_GK[i])
        for j in range(n_v):
            if (float(Ymid[j]) <= v_temp - 0.2*v_temp) or (float(Ymid[j]) >= v_temp + 0.4*v_temp):
                continue
            else:
                rmask[j,i] = 1
    wind = np.zeros([n_v,n_D])   # >>> wind-induced counts
    for i in range(9,n_D):
        v_temp = float(v_snow[i])
        for j in range(n_v):
            if float(Ymid[j]) <= v_temp - 0.7*v_temp:
                wind[j,i] = 2
    margin_fallers = np.zeros([n_v,n_D])
    for i in range(16):
        for j in range(4,n_v):
            if (float(rmask[j,i]) == 0) and (float(Ymid[j]) >= float(v_GK[i])):
                margin_fallers[j,i] = 3
        #___________________________________________________________________________
    # >>> define constants used in metrics calculation
    # >>> st dev using ip, r and s specs in wcs_fit.py
    rsigma = np.array([4.6334,0,1.29635,0.938107,0.861964,1.04708,1.36621,1.72841,2.25779,2.41439,2.69732,3.14173,
    3.66658,4.15478,4.58859,5.09409,5.64698,6.13763,6.48719,6.71595,6.8931,7.08527,7.18263,7.11314,7.23183,7.01363,
    6.91059,6.97487,7.06145,7.17577,7.102,0])
    # >>> st dev using ip spec only in wcs_fit.py
    ipsigma = np.array([0,0,1.05566,1.00748,0.897307,0.803345,0.745345,0.702103,0.747076,0.746939,0.83731,1.12322,
    1.39076,1.66773,1.94039,2.17738,2.47492,2.53875,2.43495,2.61916,2.71872,3.24875,2.9575,2.83531,2.48458,3.30476,
    3.63296,6.45171,0,0,0,0])
    ssigma = np.array([0.342592,0,0.547498,0.454442,0.418725,0.42292,0.441627,0.471931,0.528542,1.1598,1.24509,1.33774,
    1.43825,1.48046,1.54911,1.60604,1.64361,1.6879,1.66892,1.86073,1.73523,1.87607,1.83572,1.86397,1.88705,2.05337,
    2.38499,2.02697,2.37992,3.13096,4.80207,0])
    ripsigma = np.array([6.44908,0,1.3223,0.955233,0.770004,0.730101,0.854645,1.04448,1.56588,1.57923,1.52963,
    1.47653,1.54685,1.52899,1.48756,1.5625,1.6798,2.15561,1.97122,5.88769,7.18254,4.18618,0,0,0,0,0,0,0,0,0,0])
    # >>> exponent for velocities fraction in metric equation:
    metric_coeff = 3
    # >>> generate metrics m_t:
    m_r,m_ip,m_s,m_ws,m_1mm_const,m_1mm_vols,notrain = (np.zeros([n_tx]) for i in range(7))
    for timestep in range(n_tx):
        # >>> full spectrum including D<=1mm, without noise
            TEMPspec = np.copy(shiftspec[timestep,:,:])
            TEMPspec[wind == 2.], TEMPspec[margin_fallers == 3.] = 0., 0.
            if np.sum(TEMPspec[:,:]) < 30:
                continue
            else:
                if np.sum(TEMPspec[:,22:32]) >= 5:   # >>> more than 5 counts in very large bins, unlikely to be liquid precip.
                    notrain[timestep] = 1
                # >>> get volumes in small vs big D ; calculate metrics for the rest of the types
                vol_small, vol_big = np.zeros([9]), np.zeros([23])
                m_rnumi, m_ipnumi, m_snumi, m_wsnumi = (np.zeros([n_D]) for gg in range(4))  # >>> i sum in numerator of m equation
                n_cols = 0    # >>> tally of ith diam. class columns with counts aka nonzero m_t
                for i in range(n_D):
                    m_rnumj, m_ipnumj, m_snumj, m_wsnumj = (np.zeros([n_v]) for gg in range(4))   # >>> j sum in numerator of m equation
                    n_i = np.sum(TEMPspec[:,i]) # >>> sum of counts in ith D column
                    if n_i == 0:
                        continue
                    else:
                        if (float(np.sum(TEMPspec[:,9:n_D])) <= 0.05 * np.sum(TEMPspec[:,0:9])) \
                        and (np.sum(TEMPspec[:,:]) >= 30):  # >>> "small"
                            m_1mm_const[timestep] = 1.
                        # >>> figure out volume in small and big D bins, without assuming density:
                        if 0 <= i < 9:
                            vol_small[i] = n_i * D[i]**3
                        elif 9 <= i <= 32:
                            vol_big[i-9] = n_i * D[i]**3
                        # >>> now make the rest of the metrics:
                        TEMPspec[:,0:9] = 0   # >>> <1mm diam bins blanked out during phase classification # >>>
                        n_cols = n_cols + 1 # >>> tally of Diameter columns with counts
                        for j in range(n_v):   # >>> determine which myfit each bin with counts is closest to
                            N_obs = int(TEMPspec[j,i])
                            if N_obs == 0:
                                continue
                            elif wind[j,i] == 2:
                                continue
                            else:
                                v_obs = np.float(Ymid[j]) # >>> midpoint of bin with count's velocity
                                dv_r = abs(v_GK[i] - v_obs)
                                dv_ip = abs(v_ip[i] - v_obs)
                                dv_s = abs(v_snow[i] - v_obs)
                                dv_rip = abs(v_mixd[i] - v_obs)
                                m_rnumj[j] = (N_obs * (rsigma[i]/(rsigma[i] + dv_r))**metric_coeff)
                                m_ipnumj[j] = (N_obs * (ipsigma[i]/(ipsigma[i] + dv_ip))**metric_coeff)
                                m_snumj[j] = (N_obs * (ssigma[i]/(ssigma[i] + dv_s))**metric_coeff)
                                m_wsnumj[j] = (N_obs * (ripsigma[i]/(ripsigma[i] + dv_rip))**metric_coeff)

                        m_rnumi[i] = np.sum(m_rnumj[:])/n_i
                        m_ipnumi[i] = np.sum(m_ipnumj[:])/n_i
                        m_snumi[i] = np.sum(m_snumj[:])/n_i
                        m_wsnumi[i] = np.sum(m_wsnumj[:])/n_i
                if n_cols == 0:
                    continue
                else:
                    m_r[timestep] = np.sum(m_rnumi[:])/n_cols
                    m_ip[timestep] = np.sum(m_ipnumi[:])/n_cols
                    m_s[timestep] = np.sum(m_snumi[:])/n_cols
                    m_ws[timestep] = np.sum(m_wsnumi[:])/n_cols
                    if np.sum(vol_big) > 0:
                        m_1mm_vols[timestep] = np.sum(vol_small) / np.sum(vol_big)
                    elif (np.sum(vol_small) > 0) and (np.sum(vol_big) == 0):
                        m_1mm_vols[timestep] = np.sum(vol_small)
    #_____________________________________________________________________________
    #_____________________________________________________________________________
    # >>> build weather code array of len(t) using max(m_t) -- and a threshold perhaps
    my_wcs = np.zeros([n_tx])
    for i in range(n_tx):
        mt = np.array([m_r[i], m_ip[i], m_s[i], m_ws[i], m_1mm_vols[i]])
        mt_notrain = np.array([m_ip[i], m_s[i], m_ws[i]])
        if np.sum(mt) == 0:
            continue
        else:
                if max(mt) == m_r[i]:
                    my_wcs[i] = 1
                if max(mt) == m_ip[i]:
                    my_wcs[i] = 2
                if max(mt) == m_s[i]:
                    my_wcs[i] = 3
                if max(mt) == m_ws[i]:
                    my_wcs[i] = 4
                if ((m_1mm_vols[i] >= 0.01) or (np.sum(shiftspec[i,:,9:n_D]) < 5.)) and (m_1mm_const[i] == 1.):
                    my_wcs[i] = 5
                # >>> second sweep, not rain if unrealistically large diameter present
                if (notrain[i] ==1.) and (max(mt_notrain) == m_ip[i]):
                    my_wcs[i] = 2
                if (notrain[i] ==1.) and (max(mt_notrain) == m_s[i]):
                    my_wcs[i] = 3
                if (notrain[i] ==1.) and (max(mt_notrain) == m_ws[i]):
                    my_wcs[i] = 4

    # >>> re-classify potential wet snow regions
    for ii in range(n_tx):
        if (my_wcs[ii] != 4.) or (my_wcs[ii] != 2.):
            continue
        else:
            i1, i2 = ii - 3, ii + 3  # >>> encompass a full 25 minutes
            gg, hh = np.abs(i1), i2 - len(shiftspec)
            if i1 < 0.:
                i1, i2 = 0, i2 + gg
            if i2 > len(shiftspec):
                i1, i2 = ii - hh, len(shiftspec)
            typearray = my_wcs[i1:i2]
            # >>> rain & wet snow metrics are close in value and surrounding timesteps are rain
            if (len(typearray[typearray == 1.]) + len(typearray[typearray == 5.]) >= 3.) \
            and (m_ws[ii] - m_r[ii] <= 0.15):
                my_wcs[ii] = 1.
            else:
                continue
    # >>> cws_designator: choose which density correction wetsnow will get during rates calculation
    cws_designator = np.zeros([n_tx])
    for ii in range(n_tx):
        if np.abs(m_ip[ii] - m_ws[ii]) < np.abs(m_r[ii] - m_ws[ii]):
            cws_designator[ii] = 1.

   # >>> ID and flag laser beam calibration.
    DQ_FLAGS = ('missing_all_spectra', 'missing_n_of_x_spectra', 'renormalized', 'laser_calibration')
    DQ_flags, DQ_dummy = bitset('dq_flags', DQ_FLAGS), np.zeros([n_tx])
    snwh = np.array(np.where(my_wcs == 3.)[0])
    if len(snwh)!= 0 and np.all(my_wcs[0:int(snwh[0])] == 0) \
    and np.all(my_wcs[int(snwh[0])+1:int(snwh[0])+15] !=3):
        shiftspec[int(snwh[0]), :, :] = 0.
        my_wcs[int(snwh[0])], DQ_dummy[int(snwh[0])] = 0., 1.
    ipwh = np.array(np.where(my_wcs == 2.)[0])
    if len(ipwh)!= 0 and np.all(my_wcs[0:int(ipwh[0])] == 0) \
    and np.all(my_wcs[int(ipwh[0])+1:int(ipwh[0])+15] !=2):
        shiftspec[int(ipwh[0]), :, :] = 0.
        my_wcs[int(ipwh[0])], DQ_dummy[int(ipwh[0])] = 0., 1.
    # >>> add DQ flag if laser calibrated
    for ii in range(n_tx):
        if DQ_dummy[ii] == 1.:
            schon_da = DQ_array[ii]
            neu = DQ_flags(['laser_calibration'])
            DQ_array[ii] = schon_da + int(neu)

    # >>> add DQ flag if spectrum is going to be renormalized during the rates calculation
    DQ_dummy = np.zeros([n_tx])
    for i in range(n_tx):
        uncounted = np.zeros([n_D])
        unshifted = np.copy(rawspecxmin[i])
        spec = np.copy(shiftspec[i])
        if my_wcs[i] != 5.:
            continue
        else:
            for p in range(n_D):
                for pp in range(n_v):
                    if wind[pp,p] == 2:
                        uncounted[p] = uncounted[p] + unshifted[pp,p]
                    elif margin_fallers[pp,p] == 3:
                        uncounted[p] = uncounted[p] + unshifted[pp,p]
            for k in range(n_D):
                if np.sum(spec[:,k]) == 0:
                    continue
                else:
                    if ((mf_ratios[i] >= 0.5) or (w_ratios[i] >= 0.5)) and (np.sum(spec[:,k]) >= 0.3 * uncounted[k]):
                        DQ_dummy[i] = 1.
    # >>> add DQ flag
    for ii in range(n_tx):
        if DQ_dummy[ii] == 1.:
            schon_da = DQ_array[ii]
            neu = DQ_flags(['renormalized'])
            DQ_array[ii] = schon_da + int(neu)

    return(m_r, m_ip, m_s, m_ws, m_1mm_const, m_1mm_vols, my_wcs, cws_designator, DQ_array)

# In[]
# SUBROUTINE #3
# In[]
def rates_calculation(divisorx, txmin, particle_size, raw_fall_velocity, rawspecxmin, shiftspec, my_wcs, cws_designator):
    '''
    # subroutine to calculate precipitation rate for each phase of precipitation.

    Local variables for dimensions:
    ## n_t = daily total 1 minute spectra = 1440 with NANs in missing places
    ## n_tx = daily total x minute spectra
    ## n_v = 32 velocity bins
    ## n_D = 32 diameter bins
    VARIABLES INPUT:
# >>> particle_size, raw_fall_velocity (Parsivel vendor; n_v, n_D; [m/s], [mm])
# >>> rawspecxmin, shiftspec, my_wcs, cws_designator (output from my subroutines; dimension: n_tx)
    VARIABLES RETURNED:
# >>> R = weather code-specific rate array.
    ## dimension: n_tx; unit: [mm/hour]
# >>> R_r, R_ip, R_s, R_ws, R_drz = rates if every spectrum's precip were type {rain, ice pellets, snow, wet snow, drizzle}.
    ## dimension: n_tx; unit: [mm/hour]
# >>> accum = accumulation on this day, according to my algorithm
    ## dimension: n_tx; unit: [ mm ]
    ## unit conversion: rate R [mm/hour] * [1 hour/60 minutes] * spec resolution [5 mins]
# >>> vendor_accum = accumulation of this day, according to vendor alg
    ## unit: [mm]
# >>> N_n_tx = average DSDs for 5min spectra.
    ## dimension: {n_tx, n_D}; unit: [m^-3 mm^-1]
# >>> N_dayavg = average DSD for this day.
    ## dimension: n_D; unit: [m^-3 mm^-1]
    '''
    import numpy as np
    #______________________________________________________________________________
        # >>> GET LOCAL VARIABLES AGAIN
    n_tx = len(txmin)  # >>> total # of 1-min intervals in present dataset
    n_v = len(raw_fall_velocity)   # >>> 32 raw_fall_velocity bins
    n_D = len(particle_size)   # >>> 32 particle_size bins
    X, Y = np.meshgrid(particle_size, raw_fall_velocity)
    # >>> midpoints of vel bins (Ymid) and size bins (D)
    x, y = X[0,:], Y[:,0]
    D = np.array((x[1:] + x[:-1]) / 2).tolist()
    D.insert(0, x[0]/2)
    D = np.array(D[:])
    Ymid = np.array((y[1:] + y[:-1]) / 2).tolist()
    Ymid.insert(0, y[0]/2)
    Ymid = np.array(Ymid[:])
    # >>> my fits: phase-specific expressions for fall velocities
    a_ip, a_s, a_rip = 2.47563950652,  1.29129876915, -13.9193270007
    b_ip, b_s, b_rip = 0.249683670041, 0.352624015539, -0.34355822607
    c_rip = -1.78307830508
    # >>> fall velocity equations
    v_snow = a_s * D ** b_s
    v_ip = a_ip * D ** b_ip
    v_mixd = c_rip - a_rip * np.exp(-D ** b_rip)
    v_GK = 9.65 - 10.3 * np.e**(-0.6 * D)  # >>> rain fall speed equation used thruout
       #____________________________________________________________________________
    # >>> MASKING WINDY AND MARGIN FALLERS REGIONS
    rmask = np.zeros([n_v,n_D])   # >>> rain mask
    for i in range(25):
        v_temp = float(v_GK[i])
        for j in range(n_v):
            if (float(Ymid[j]) <= v_temp - 0.2*v_temp) or (float(Ymid[j]) >= v_temp + 0.4*v_temp):
                continue
            else:
                rmask[j,i] = 1
    wind = np.zeros([n_v,n_D])   # >>> wind-induced counts
    for i in range(9,n_D):
        v_temp = float(v_snow[i])
        for j in range(n_v):
            if float(Ymid[j]) <= v_temp - 0.7*v_temp:
                wind[j,i] = 2
    margin_fallers = np.zeros([n_v,n_D])
    for i in range(16):
        for j in range(4,n_v):
            if (float(rmask[j,i]) == 0) and (float(Ymid[j]) >= float(v_GK[i])):
                margin_fallers[j,i] = 3
    #____________________________________________________________________________
    # >>> RATIOS OF COUNTS IN MARGIN FALLERS, WINDY REGIONS
    mf_ratios = np.zeros(n_tx)
    w_ratios = np.zeros(n_tx)
    for k in range(n_tx):
        spec = np.copy(rawspecxmin[k,:,:])
        if np.sum(spec) == 0:
            continue
        else:
            mf_dummy = np.zeros([n_v,n_D])
            w_dummy = np.zeros([n_v,n_D])
            for i in range(n_D):
                for j in range(n_v):
                    if margin_fallers[j,i] == 3:
                        mf_dummy[j,i] = spec[j,i]
                    elif wind[j,i] == 2:
                        w_dummy[j,i] = spec[j,i]
            mf_ratios[k] = float(np.sum(mf_dummy)/np.sum(spec))
            w_ratios[k] = float(np.sum(w_dummy)/np.sum(spec))
    #________________________________________________________________________________
    # >>> wet snow density correction employs cws_designator from metrics subroutine!
    # >>> wet snow density correction for more liquid precip:
    cwsL = np.zeros(len(D))
    for k in range(len(D)):
        if k <= 8:
            cwsL[k] = 1
        else:
            cwsL[k] = ((v_mixd[k]/v_snow[k])**2)*( 0.178 * D[k] ** -0.922)/(0.977 + 0.178 * D[k] ** -0.922)
    # >>> wet snow density correction for more frozen precip:
    cwsF = np.zeros(len(D))
    cwsF[0:2] = 1.
    for k in range(2,len(D)):
        cwsF[k] = ((v_mixd[k]/(3*v_snow[k]))**(1/3))*( 0.178 * D[k] ** -0.922)/(0.977 + 0.178 * D[k] ** -0.922)
    # >>> snow density correction:
    cs = np.zeros(len(D))
    for k in range(len(D)):
        cs[k] = ( 0.178 * D[k] ** -0.922)/(0.977 + 0.178 * D[k] ** -0.922)
        if k >= 20:
            cs[k] = 0.01 * ( 0.178 * D[k] ** -0.922)/(0.977 + 0.178 * D[k] ** -0.922)
    # >>> ice density correction:
    ci = (0.977 - .934)/.997
    #________________________________________________________________________________
    # >>> key variables for rates calculation:
    # >>> drop concentration N = [ 1 / (S * dD * dt) ] sum(C_vi / V_i)
    # >>> dD = bin width; dt = 1 min = (1/60) hours; C_vi = counts in bin (v = i:i+1);
    # >>> V_i = ith bin vel
    L, B, dt = 0.180, 0.030, 60*divisorx # >>> [m]; beam length, [m]; beam width, seconds in x mins
    dD = [x[0]] # >>> spacing of diameter bins
    dD.extend(np.diff(x).tolist())
    dD = np.array(dD)
    #________________________________________________________________________________
    # >>> precip rates, accumulation:
    R = np.zeros([n_tx])    # >>> for each timestep, weather code-specific precip rate
    # >>> for each timestep, as if all spectra were rain, ice, etc.
    R_r, R_ip, R_s, R_ws, R_drz, D_eff_tx = (np.zeros([n_tx]) for p in range(6))
    N_n_tx = np.zeros([n_tx, n_D])
    for i in range(n_tx):
        if my_wcs[i] == 0:
            continue
        else:
            uncounted = np.zeros([n_D])
            unshifted = np.copy(rawspecxmin[i])
            spec = np.copy(shiftspec[i])
            if my_wcs[i] == 1.:
                spec[:,21:] = 0.
            for p in range(n_D):
                for pp in range(n_v):
                    if wind[pp,p] == 2:
                        uncounted[p] = uncounted[p] + unshifted[pp,p]
                    elif margin_fallers[pp,p] == 3:
                        uncounted[p] = uncounted[p] + unshifted[pp,p]
            N_kpars, R_rk, R_ipk, R_sk, R_wsk, R_drzk, R_k = (np.zeros([n_D]) for gg in range(7))
            for k in range(n_D):
                Cjk_Vj = np.zeros([n_D])
                if np.sum(spec[:,k]) == 0:
                    continue
                else:
                    for j in range(n_v):
                        C_jk = spec[j,k]
                        if C_jk == 0:
                            continue
                        else:
                            V_j = Ymid[j]
                            Cjk_Vj[j] = C_jk / V_j
                    # >>> renormalize spectrum using 50% of neglected counts in noise region
                    if ((mf_ratios[i] >= 0.5) or (w_ratios[i] >= 0.5)) and (np.sum(spec[:,k]) >= 0.3 * uncounted[k]) \
                    and (my_wcs[i]==5.):
                        Cjk_Vj[Cjk_Vj == np.median(Cjk_Vj)] = Cjk_Vj[Cjk_Vj == np.median(Cjk_Vj)] \
                        + 0.5 * uncounted[k] / Ymid[Cjk_Vj == np.median(Cjk_Vj)]
                    C_V_sum = np.sum(Cjk_Vj) # >>> [m/s]^-1
                    S = L * ( B - ( (D[k] * 10**-3) / 2 ) ) # >>> [m^2]
                    N_kpars[k] = (1 / (S * dD[k] * dt)) * C_V_sum # >>> [m^2*mm*s]^-1 * [m/s]^-1 = [m^-3 mm^-1]
                    R_rk[k] = N_kpars[k] * v_GK[k] * D[k]**3
                    R_ipk[k] = N_kpars[k] * v_ip[k] * D[k]**3 * ci
                    R_sk[k] = N_kpars[k] * v_snow[k] * D[k]**3 * cs[k]  # >>> Zhang et al (2011) snow density
                    if cws_designator[i] == 1.:
                        cws = np.copy(cwsF)
                    if cws_designator[i] == 0.:
                        cws = np.copy(cwsL)
                    R_wsk[k] = N_kpars[k] * v_mixd[k] * D[k]**3  * cws[k]
                    if my_wcs[i] == 1 or my_wcs[i] == 5:
                        R_k[k] = np.copy(R_rk[k])
                    elif my_wcs[i] == 2:
                        R_k[k] = np.copy(R_ipk[k])
                    elif my_wcs[i] == 3:
                        R_k[k] = np.copy(R_sk[k])
                    elif my_wcs[i] == 4:
                        R_k[k] = np.copy(R_wsk[k])
                    if my_wcs[i] == 5: # >>> array includes rates for small particle spectra ONLY.
                        R_drzk[k] = np.copy(R_rk[k])
                    else:
                        continue
            N_n_tx[i,:] = N_kpars[:]
            D_eff_tx[i] = np.nansum(N_kpars[:] * D[:]**3 * dD[:]) / np.nansum(N_kpars[:] * D[:]**2 * dD[:])
            R[i] = np.trapz(R_k,x=D,axis=0) * (6 * np.pi) * (10**-4)
            R_r[i] = np.trapz(R_rk,x=D,axis=0) * (6 * np.pi) * (10**-4)
            R_ip[i] = np.trapz(R_ipk,x=D,axis=0) * (6 * np.pi) * (10**-4)
            R_s[i] = np.trapz(R_sk,x=D,axis=0) * (6 * np.pi) * (10**-4)
            R_ws[i] = np.trapz(R_wsk,x=D,axis=0) * (6 * np.pi) * (10**-4)
            R_drz[i] = np.trapz(R_drzk,x=D,axis=0) * (6 * np.pi) * (10**-4)

    # >>> effective radius in microns; average DSD from this day
    r_eff_tx = 1000 * (D_eff_tx/2)   # >>> units: microns
    # >>> average DSD across n_tx dimension
    N_dayavg = np.mean(N_n_tx, axis=0)

    # >>> using modes, change 'small' to reflect phase if it's close to frozen spectra.
    for ii in range(len(my_wcs)):
        i1, i2 = ii - 2 * divisorx, ii + 2 * divisorx
        if i1 < 0:
            i2 = i2 + abs(i1)
            i1 = 0
        if i2 > len(my_wcs):
            i2 = len(my_wcs)
        type_tmp = np.copy(my_wcs[i1:i2])
        type_max = max([len(type_tmp[type_tmp==2.]),len(type_tmp[type_tmp==3.]),len(type_tmp[type_tmp==4.])])
        if (my_wcs[ii]==5.) and (len(type_tmp[(type_tmp!=1.) & (type_tmp!=0.)]) >= len(type_tmp[type_tmp==1.])):
            if type_max==len(type_tmp[type_tmp==2.]):
                R[ii] = R_ip[ii]
            elif type_max==len(type_tmp[type_tmp==3.]):
                R[ii] = R_s[ii]
            elif type_max==len(type_tmp[type_tmp==4.]):
                R[ii] = R_ws[ii]
                if (len(my_wcs[my_wcs==2.]) >= len(my_wcs[my_wcs==4.])):
                    R[ii] = R_ip[ii]
            elif (type_max==0.) and (len(my_wcs[my_wcs==2.]) > 0.) and \
            (len(my_wcs[my_wcs==2.]) >= len(my_wcs[my_wcs==4.])):
                R[ii] = R_ip[ii]
            else:
                continue

    # >>> get rid of unrealistic rates
    dummyR = np.copy(R)
    dummyR[dummyR == 0.] = np.nan
    R_act_trckr = np.zeros([len(R)])
    for idx in range(len(R)):
        if ((my_wcs[idx] == 0.) or (np.std(shiftspec[idx]) > 3. * np.std(shiftspec))) and (R[idx] <= np.median(R[my_wcs==my_wcs[idx]])):
            R_act_trckr[idx] = 1.
            continue
        else:
            if (my_wcs[idx] == 1.) and ((R[idx] >= 20.) or (R[idx] >= np.median(R[my_wcs==1.])+4*np.std(R[my_wcs==1.]))):
                R[idx], R_act_trckr[idx] = np.median(R[my_wcs==1.]), 2.
                if (len(my_wcs[my_wcs==1]) <= 3.) and (50. * np.median(R[my_wcs==1.])):
                    R[idx], R_act_trckr[idx] = 0., 2.5
            if (my_wcs[idx]==5) and ((R[idx] >= 8.) or (R[idx] >= np.median(R[my_wcs==5.]) + 4.*np.std(R[my_wcs==5.]))):
                R[idx], R_act_trckr[idx] = np.median(R[my_wcs==5.]), 4.
            if (my_wcs[idx] == 2.) and (R[idx] >= np.median(R[my_wcs==2.])+4.*np.std(R[my_wcs==2.])):
                R[idx], R_act_trckr[idx] = np.median(R[my_wcs==2.]), 6.
            if (my_wcs[idx]==4.) and ((R[idx] >= np.median(R[my_wcs==4.]) + 4.*np.std(R[my_wcs==4.])) or (R[idx] >= 50.*np.median(R[my_wcs==4.]))):
                R[idx], R_act_trckr[idx] = np.median(R[my_wcs==4.]), 7.
            if (my_wcs[idx] == 3.) and ((R[idx] >= np.median(R[my_wcs==3.])+4.*np.std(R[my_wcs==3.])) or (R[idx] >= 7.)):
                R[idx], R_act_trckr[idx] = np.median(R[my_wcs==3.]), 8.
#
    return(R, R_r, R_ip, R_s, R_ws, R_drz, N_n_tx, r_eff_tx, N_dayavg, R_act_trckr)

# In[]
def scale_spectra(shiftspec, my_wcs, amp_missing, divisorx, w_ratios, DQ_array):
    '''
    - Depending on laser beam amplitude and particle phase, multiply velocity-diameter histograms by various amounts to increase counts.
    - Do not scale spectra that already have anomalously high counts (statistically predetermined for each phase using 9 months of data)
    - Do not scale spectra of type 'small' if they have already been DQ flagged for renormalization
    - Do not scale spectra with high counts in wind noise region
        Local variables for dimensions:
    ## n_t = daily total 1 minute spectra = 1440 with NANs in missing places
    ## n_tx = daily total x minute spectra
    ## n_v = 32 velocity bins
    ## n_D = 32 diameter bins
    VARIABLES INPUT:
# >>> amp_missing (output from my subroutines; dimension: n_t; unit: [arb. counts])
# >>> shiftspec, my_wcs, w_ratios, mf_ratios, DQ_array (output from my subroutines; dimension: n_tx)
    VARIABLES RETURNED:
# >>> shiftspec_scaled -- same spectrum as shiftspec but with more counts.
      ## dimension: n_tx; unit: [counts]
    '''
#    import scipy.signal as signal
    import numpy as np
    # >>> Laser amplitude Buterworth filter
#    N, Wn  = 3, 0.05    # Filter order, cutoff frequency
#    B, A = signal.butter(N, Wn, output='ba')
#    ampsm = signal.filtfilt(B, A, amp_missing) # >>> amplitude smoothed with filter
    ampsm = running_mean(amp_missing, 10)
    ampsm = ampsm.tolist()
    ampsm.insert(3, ampsm[3]), ampsm.insert(2, ampsm[2]), ampsm.insert(1, ampsm[1])
    ampsm.insert(0, ampsm[0]), ampsm.insert(len(ampsm)-5, ampsm[-5])
    ampsm.insert(len(ampsm)-4, ampsm[-4]), ampsm.insert(len(ampsm)-3, ampsm[-3])
    ampsm.insert(len(ampsm)-2, ampsm[-2]), ampsm.insert(len(ampsm)-1, ampsm[-1])
    sm_alpha, sm_amps = np.array([8.304, 7.62, 9.135, 11.016, 3.171]), np.array([6233.0, 8596.0, 10870.0, 13151.0, 16139.0])
    r_alpha, r_amps = np.array([3.387, 5.113, 2.141, 2.23, 1.]), np.array([ 5508.,  8246., 10332., 12143., 15921.])
    ip_alpha, ip_amps = np.array([2.674, 4.274, 1.483, 6.936, 5.0]), np.array([5550., 7910., 9765., 11675., 15904.])
    s_alpha, s_amps = np.array([1., 1.857, 1.147, 2.494, 1.]), np.array([4966.0, 7395.0, 9320.0, 11734.0, 14138.0])
    ws_alpha, ws_amps = 0.5 * np.array([2.417, 3.442, 3.8, 5.529, 1.]), np.array([5650.0, 8381.0, 10487.0, 12804.0, 16190.0])

    # >>> statistics on median, st dev of counts in spectra by type:
    r_med, r_std = 69., 81.
    ip_med, ip_std = 120., 278.
    s_med, s_std = 175., 243.
    ws_med, ws_std = 57., 137.
    sm_med, sm_std = 35., 52.

    shiftspec_scaled = np.copy(shiftspec)
    for ii in range(len(my_wcs)):
        if (my_wcs[ii]==0.):
            continue
        else:
            i1 = ii*divisorx
            i2 = i1 + divisorx
            if np.isnan(np.sum(ampsm[i1:i2])):
                ampsm[i1:i2] = amp_missing[i1:i2]
            amp_med = np.median(ampsm[i1:i2])
            if np.isnan(amp_med):
                continue
            else:
                j2, j1 = ii + 3, ii - 3
                if j1 < 0:
                    j2 = j2 + np.abs(j1)
                    j1 = 0
                if j2 > len(my_wcs):
                    j1 = j1 - (j2 - len(my_wcs))
                    j2 = len(my_wcs)
                typewindow = np.copy(my_wcs[j1:j2])
                if (my_wcs[ii]==1 and np.nansum(shiftspec[ii]/divisorx) >= 3.*r_std + r_med)  or \
                (my_wcs[ii]==2 and np.nansum(shiftspec[ii]/divisorx) >= 3.*ip_std + ip_med) or \
                (my_wcs[ii]==3 and np.nansum(shiftspec[ii]/divisorx) >= 3.*s_std + s_med) or \
                (my_wcs[ii]==4 and np.nansum(shiftspec[ii]/divisorx) >= 3.*ws_std + ws_med) or \
                (my_wcs[ii]==5) and ((np.nansum(shiftspec[ii]/divisorx) >= 3.*sm_std + sm_med) or \
                                     (DQ_array[ii]==4.)):
    #                print('filter 1')
                    continue
                elif (amp_med >= 10000.) and \
                ((my_wcs[ii]==1 and np.nansum(shiftspec[ii]/divisorx) >= 2.*r_std + r_med)  or \
                (my_wcs[ii]==2 and np.nansum(shiftspec[ii]/divisorx) >= 2.*ip_std + ip_med) or \
                (my_wcs[ii]==3 and np.nansum(shiftspec[ii]/divisorx) >= 2.*s_std + s_med) or \
                (my_wcs[ii]==4 and np.nansum(shiftspec[ii]/divisorx) >= 2.*ws_std + ws_med)) or \
                (my_wcs[ii]==5) and ((np.nansum(shiftspec[ii]/divisorx) >= 2.*sm_std + sm_med) or \
                                     (DQ_array[ii]==4.)):
    #                print('filter 2')
                    continue
                elif ((w_ratios[ii]>=.05) and (len(typewindow[typewindow==5.])+len(typewindow[typewindow==2])+ \
                    len(typewindow[typewindow==4]) > 2.)):
    #                print('filter 3')
                    continue
                elif (len(my_wcs[(my_wcs!=0.) & (my_wcs!=1.) & (my_wcs!=5.)] >= 3.)) and \
                (np.std(shiftspec[ii])>=1.5+np.std(shiftspec)):
    #                print('filter 4')
                    continue
                else:
                    if my_wcs[ii]==5.:
                        amp_near = find_nearest(sm_amps, amp_med)
                        alpha_near = sm_alpha[sm_amps==amp_near]
                    elif my_wcs[ii]==1.:
                        amp_near = find_nearest(r_amps, amp_med)
                        alpha_near = r_alpha[r_amps==amp_near]
                    elif my_wcs[ii]==2.:
                        amp_near = find_nearest(ip_amps, amp_med)
                        alpha_near = ip_alpha[ip_amps==amp_near]
                    elif my_wcs[ii]==3.:
                        amp_near = find_nearest(s_amps, amp_med)
                        alpha_near = s_alpha[s_amps==amp_near]
                    elif my_wcs[ii]==4.:
                        amp_near = find_nearest(ws_amps, amp_med)
                        alpha_near = ws_alpha[ws_amps==amp_near]
                    shiftspec_scaled[ii] = alpha_near * shiftspec_scaled[ii]

    return(shiftspec_scaled)
#
#    # >>> statistics on median, st dev of counts in spectra by type:
#    r_med, r_std = 69., 81.
#    ip_med, ip_std = 120., 278.
#    s_med, s_std = 175., 243.
#    ws_med, ws_std = 57., 137.
#    sm_med, sm_std = 35., 52.
#
#    for ii in range(len(my_wcs)):
#        i1 = ii*divisorx
#        i2 = i1 + divisorx
#        if np.isnan(np.sum(ampsm[i1:i2])):
#            ampsm[i1:i2] = amp_missing[i1:i2]
#        j2, j1 = ii + 3, ii - 3
#        if j1 < 0:
#            j2 = j2 + np.abs(j1)
#            j1 = 0
#        if j2 > len(my_wcs):
#            j1 = j1 - (j2 - len(my_wcs))
#            j2 = len(my_wcs)
#        typewindow = np.copy(my_wcs[j1:j2])
#        if (my_wcs[ii]==1 and np.nansum(shiftspec[ii]/divisorx) >= 3.*r_std + r_med)  or \
#        (my_wcs[ii]==2 and np.nansum(shiftspec[ii]/divisorx) >= 3.*ip_std + ip_med) or \
#        (my_wcs[ii]==3 and np.nansum(shiftspec[ii]/divisorx) >= 3.*s_std + s_med) or \
#        (my_wcs[ii]==4 and np.nansum(shiftspec[ii]/divisorx) >= 3.*ws_std + ws_med) or \
#        (my_wcs[ii]==5) and ((np.nansum(shiftspec[ii]/divisorx) >= 4.*sm_std + sm_med) or \
#                             (DQ_array[ii]==4.)):
#            continue
#        elif (np.mean(ampsm[i1:i2]) >= 10000.) and \
#        ((my_wcs[ii]==1 and np.nansum(shiftspec[ii]/divisorx) >= 3.*r_std + r_med)  or \
#        (my_wcs[ii]==2 and np.nansum(shiftspec[ii]/divisorx) >= 2.*ip_std + ip_med) or \
#        (my_wcs[ii]==3 and np.nansum(shiftspec[ii]/divisorx) >= 2.*s_std + s_med) or \
#        (my_wcs[ii]==4 and np.nansum(shiftspec[ii]/divisorx) >= 2.*ws_std + ws_med)):
#            continue
#        elif (w_ratios[ii]>=.05) and (len(typewindow[typewindow==5.])+len(typewindow[typewindow==2])+ \
#            len(typewindow[typewindow==4]) > 2.):
#            continue
#        else:
#            if 16000 < np.median(ampsm[i1:i2]):
#                # >>> RAIN SPECTRA
#                if my_wcs[ii] == 1:
#                    shiftspec_scaled[ii,:,7:13] = 1.5 * shiftspec_scaled[ii,:,7:13]
#                    if (len(typewindow[typewindow==1.])+len(typewindow[typewindow==5]) >= 6.) and \
#                    (len(typewindow[typewindow==2])+len(typewindow[typewindow==3])+len(typewindow[typewindow==4])==0.):
#                        shiftspec_scaled[ii,:,7:13] = 2. * shiftspec[ii,:,7:13]
#                # >>> SMALL SPECTRA
#                if (my_wcs[ii] == 5.):
#                    shiftspec_scaled[ii] = 3. * shiftspec_scaled[ii]
#                # >>> ALL OTHERS
#                if (1 < my_wcs[ii] < 5.):
#                    shiftspec_scaled[ii, :, 7:13] = 1.5 * shiftspec_scaled[ii,:,7:13]
#            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#            if 11000 < np.median(ampsm[i1:i2]) <= 16000:
#                # >>> RAIN SPECTRA
#                if my_wcs[ii] == 1:
#                    shiftspec_scaled[ii,:,7:13] = 2 * shiftspec_scaled[ii,:,7:13]
#                    if (len(typewindow[typewindow==1.])+len(typewindow[typewindow==5]) >= 6.) and \
#                    (len(typewindow[typewindow==2])+len(typewindow[typewindow==3])+len(typewindow[typewindow==4])==0.):
#                        shiftspec_scaled[ii,:,7:13] = 2.5 * shiftspec[ii,:,7:13]
#                # >>> SMALL SPECTRA
#                if (my_wcs[ii] == 5.):
#                    shiftspec_scaled[ii] = 4. * shiftspec_scaled[ii]
#                # >>> ALL OTHERS
#                if (1 < my_wcs[ii] < 5.):
#                    shiftspec_scaled[ii, :, 7:13] = 1.5 * shiftspec_scaled[ii,:,7:13]
#            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#            if 9000 <= np.median(ampsm[i1:i2]) <= 11000.:
#                # >>> RAIN SPECTRA
#                if (my_wcs[ii] == 1):
#                    shiftspec_scaled[ii,:,7:13] = 4 * shiftspec_scaled[ii,:,7:13]
#                    if (len(typewindow[typewindow==1.])+len(typewindow[typewindow==5]) >= 4.) and \
#                    (len(typewindow[typewindow==2])+len(typewindow[typewindow==3])+len(typewindow[typewindow==4])==0.):
#                        shiftspec_scaled[ii,:,7:13] = 6. * shiftspec[ii,:,7:13]
#                # >>> SMALL SPECTRA
#                if (my_wcs[ii] == 5.):
#                    shiftspec_scaled[ii] = 5. * shiftspec_scaled[ii]
#
#                # >>> SNOW
#                if (my_wcs[ii] == 3.):
#                    shiftspec_scaled[ii,:,7:13] = 1.5 * shiftspec_scaled[ii,:,7:13]
#                # >>> ALL OTHERS
#                if (my_wcs[ii] == 2.) or (my_wcs[ii] == 4.):
#                    shiftspec_scaled[ii,:,7:13] = 4. * shiftspec_scaled[ii,:,7:13]
#            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#            if 7500 <= np.median(ampsm[i1:i2]) < 9000.:
#                # >>> RAIN SPECTRA
#                if (my_wcs[ii] == 1):
#                    shiftspec_scaled[ii,:,7:13] = 7. * shiftspec_scaled[ii,:,7:13]
#
#                # >>> SMALL SPECTRA
#                if (my_wcs[ii] == 5.):# and (mf_ratios[ii]<=0.17):
#                    shiftspec_scaled[ii] = 7. * shiftspec_scaled[ii]
#
#                # >>> SNOW
#                if (my_wcs[ii] == 3.):
#                    shiftspec_scaled[ii,:,7:13] = 2.0 * shiftspec_scaled[ii,:,7:13]
#                # >>> ALL OTHERS
#                if (my_wcs[ii] == 2.) or (my_wcs[ii] == 4.):
#                    shiftspec_scaled[ii,:,7:13] = 6. * shiftspec_scaled[ii,:,7:13]
#            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#            if 5000 <= np.median(ampsm[i1:i2]) < 7500.:
#                # >>> RAIN SPECTRA
#                if (my_wcs[ii] == 1):
#                    shiftspec_scaled[ii,:,7:13] = 7. * shiftspec_scaled[ii,:,7:13]
#                    if (len(typewindow[typewindow==1.])+len(typewindow[typewindow==5]) >= 4.):
#                        shiftspec_scaled[ii,:,7:13] = 11. * shiftspec[ii,:,7:13]
#                # >>> SMALL SPECTRA
#                if (my_wcs[ii] == 5.):
#                    shiftspec_scaled[ii] = 7. * shiftspec_scaled[ii]
#                # >>> SNOW
#                if (my_wcs[ii] == 3.):
#                    shiftspec_scaled[ii,:,7:13] = 1.5 * shiftspec_scaled[ii,:,7:13]
#                # >>> ALL OTHERS
#                if (my_wcs[ii] == 2.) or (my_wcs[ii] == 4.):
#                    shiftspec_scaled[ii,:,7:13] = 8. * shiftspec_scaled[ii,:,7:13]
#            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#            if 5000 > np.median(ampsm[i1:i2]):
#                # >>> RAIN SPECTRA
#                if my_wcs[ii] == 1:
#                    shiftspec_scaled[ii,:,7:13] = 14. * shiftspec_scaled[ii,:,7:13]
#                    if (len(typewindow[typewindow==1.])+len(typewindow[typewindow==5]) >= 4.) and \
#                    (len(typewindow[typewindow==2])+len(typewindow[typewindow==3])+len(typewindow[typewindow==4])==0.):
#                        shiftspec_scaled[ii,:,7:13] = 15. * shiftspec[ii,:,7:13]
#                # >>> SMALL SPECTRA
#                if (my_wcs[ii] == 5.):
#                    shiftspec_scaled[ii] = 8. * shiftspec_scaled[ii]
#                    if (len(typewindow[typewindow==1.])+len(typewindow[typewindow==5]) >= 4.) and \
#                    (len(typewindow[typewindow==2])+len(typewindow[typewindow==3])+len(typewindow[typewindow==4])==0.):
#                        shiftspec_scaled[ii] = 9. * shiftspec[ii]
#                # >>> ALL OTHERS
#                if (1 < my_wcs[ii] < 5.):
#                    shiftspec_scaled[ii,:,7:13] = 12. * shiftspec_scaled[ii,:,7:13]
#
#    return(shiftspec_scaled)

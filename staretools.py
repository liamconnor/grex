import os

import time
import numpy as np 
from sigpyproc.Readers import FilReader
import pandas as pd
from astropy.time import Time
from astropy import units as u

sgr_1935_ra = 293.731999*u.deg
dm_sgr1935 = 332.

def get_RA_overhead(mjd=None,longitude_rad=-2.064427799136453):
    if mjd is None:
        t = Time.now()
    elif mjd > 40000:
        t = Time(mjd, format='mjd')
    else:
        print("Expecting MJD as float >500000 or None")
        return

    t.delta_ut1_utc = 0.
    ra = t.sidereal_time('mean', longitude_rad*u.rad)
    return ra

def get_RA_diff(mjd, src_RA):
    ra = get_RA_overhead(mjd)
    return ra-src_RA

def get_mjd_cand(fncand):
    f = open(fncand,'r')

    mjd_arr = []
    for line in f:
        ll = line.split(' ')
        try:
            mjd = np.float(ll[8])+(np.float(ll[9])+np.float(ll[10])/60.+np.float(ll[11])/3600.0)/24.0
        except:
            mjd = -1
        mjd_arr.append(mjd)

    return np.array(mjd_arr)

def get_mjd_cand_pd(data):
    mjd = data['mjd_day']+(data['mjd_hr']+data['mjd_min']/60.+data['mjd_sec']/3600.)/24.

    return mjd

def read_heim_pandas(fncand, skiprows=0):
    if fncand.endswith('csv'):
        data = pd.read_csv(fncand)
        return data
    elif fncand.endswith('cand'):

        data = pd.read_table(fncand,
                      skiprows=skiprows, delimiter=' ', 
                      header=None, usecols=range(12))

        heim_columns = ['snr','cand','time_sec',
                    'log2width','unknown2','dm',
                    'unknown3','mjdx','mjd_day',
                    'mjd_hr','mjd_min','mjd_sec']

        data.columns = heim_columns
        return data
    else:
        print("Expected a .csv or .cand file.")
        return 

def rsync_heimdall_cand():
    os.system("rsync -avzhe ssh user@158.154.14.10:/home/user/cand_times_sync/ /home/user/cand_times_sync");
    os.system("rsync -avzhe ssh user@166.140.120.248:/home/user/cand_times_sync/ /home/user/cand_times_sync_od");
    time.sleep(60)

def check_latest_cand_time():
    flist = glob.glob('/home/user/candidates/*.fil')
    flmax = max(flist, key=os.path.getctime)
    time_since_cand = os.path.getctime(flmax)-time.time()
    return time_since_cand

def read_fil_data_stare(fn, start=0, stop=1):
    """ Read in filterbank data"""
    fil_obj = FilReader(fn)
    header = fil_obj.header
    delta_t = header['tsamp'] # delta_t in seconds                                
    fch1 = header['fch1']
    nchans = header['nchans']
    foff = header['foff']
    fch_f = fch1 + nchans*foff
    freq = np.linspace(fch1,fch_f,nchans)
    try:
        data = fil_obj.readBlock(start, -1)
    except(ValueError):
        data = fil_obj.readBlock(start, header['nsamples'])
    return data, freq, delta_t, header








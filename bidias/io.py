
import numpy as np

import pandas as pd

from tfer import tfer
from autils import autils, props
from bidias import Grid

fd = '..\\bidias\\data\\24bbcan\\cpma_sp2_20240715'
fn1 = fd + '\\' + 'cpma\\' + '20240715_TEST000.cpma'

fn2 = fd + '\\' + 'sp2\\20240715152133\\' + 'SP2XR_PbP_20240715152133_x0001.csv'


def read_cpma(fn):
    df = pd.read_csv(fn, sep='\t', skiprows=(np.arange(0,7)))
    df = df.loc[np.arange(0, np.flatnonzero(df['Datum#'] == 'END OF SCAN')[0])]
    return df

def read_cp2(fn1, fn2):
    # Read CPMA file.
    df1 = read_cpma(fn1)
    t1 = pd.to_datetime(df1['Time']).dt.time
    m_star = np.asarray(df1['Mass (fg)']).astype(np.float64)

    # Read SP2 file.
    df2 = pd.read_csv(fn2)
    mrbc = df2['Incand Mass (fg)']
    mrbc = mrbc[~(mrbc == 0.06)]
    t2 = pd.to_datetime(df2['Time Stamp (sec)'], unit='D', origin='1899-12-30').dt.time

    # Collect particles by CPMA setpoints.
    particles = [[]] * (len(t1) - 1)
    for ii in range(1, len(t1)):
        particles[ii-1] = mrbc[np.logical_and(t2 > t1[ii - 1], t2 < t1[ii])]

    nrbc = 80
    b = np.ones((len(t1) - 1, nrbc))
    mrbc_star = np.logspace(np.log10( np.min(mrbc)), np.log10(np.max(mrbc)), nrbc + 1)
    for ii in range(1, len(t1) - 1):
        b[ii-1, :], _ = np.histogram(particles[ii], mrbc_star)
    mrbc_star = np.exp((np.log(mrbc_star[1:]) + np.log(mrbc_star[0:-1])) / 2)
    b = b.T.ravel()  # vectorized data

    # Properties for creating setpoints.
    prop_pma = props.pma()
    prop_pma = autils.massmob_add(prop_pma, 'water')
    props.update_flow(prop_pma, 0.3 * 0.0000166667)

    sp, _ = tfer.get_setpoint(prop_pma, 'V', df1['Voltage (V)'], 'omega', df1['Speed (rad/s)'])
    # spa, _ = tfer.get_setpoint(prop_pma, 'm_star', m_star * 1e-18, 'Rm', [5])
    
    sp = sp[0:-1]

    # Create grid corresponding to data.
    grid_b = Grid.Grid(edges=[tfer.unpack(sp)['m_star'] * 1e18, mrbc_star])
    grid_b.type = ['mp', 'mrbc']

    return b, grid_b, sp, prop_pma


def read_smps(fn):
    """
    Reads a TSI SMPS .txt or .csv file exported from AIM software.
    
    Parameters:
        file_path (str): Path to the SMPS file.
    
    Returns:
        tuple: (metadata dict, dataframe with size distribution)
    """
    with open(fn, 'r', encoding='cp1252') as f:
        lines = f.readlines()

    # Identify relevant metadata.
    prop = {}
    def parse_meta(input_str):
        return np.array([float(input_str.split('\t', 1)[1].replace('\n', ''))])
    def parse_flows(input_str):
        Q = line.split('\t')
        Q[-1].replace('\n', '')  # remove trailing line break
        Q = Q[1:]  # remove text label
        Q = np.array(Q, dtype=float)  # convert to array of floats
        if np.all(Q == Q[0]):
            Q = np.array([Q[0]])
        return Q
    
    for ii, line in enumerate(lines):
        if "Inner Radius" in line:
            prop['R1'] = parse_meta(line) / 100  # plus, convert from cm to m
        elif "Outer Radius" in line:
            prop['R2'] = parse_meta(line) / 100
        elif "Length" in line:
            prop['L'] = parse_meta(line) / 100
        elif "Classifier Model" in line:
            prop['model'] = line.split('\t', 1)[1].replace('\n', '')
        elif "Reference Gas Temperature" in line:
            prop['T'] = parse_meta(line)
        elif "Reference Gas Pressure" in line:
            prop['p'] = parse_meta(line) / 101.3
        elif "Sheath Flow" in line:
            prop['Qc'] = parse_flows(line) / 60000  # plus, convert from lpm to m3/s
            prop['Qm'] = prop['Qc']  # assume equal flow
        elif "Aerosol Flow" in line:
            prop['Qa'] = parse_flows(line) / 60000  # aerosol flow
            prop['Qs'] = prop['Qa']  # assume equal flow (sample flow)

    # Identify start of data.
    data_start = 0
    for ii, line in enumerate(lines):
        if "Diameter" in line:
            data_start = ii
            break

    # Identify end of data.
    data_end = 0
    for ii, line in enumerate(lines):
        if "Scan" in line:
            data_end = ii
            break

    # Number of rows occupied by data.
    n_data_rows = data_end - data_start - 1

    # Read numerical data into DataFrame
    df = pd.read_csv(fn, skiprows=data_start+1, nrows=n_data_rows, encoding='cp1252', sep='\t', header=None)
    data = np.asarray(df.loc[:,1:])
    dm = np.asarray(df.loc[:,0])
    return data, dm, prop
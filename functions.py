# Basic comp. tools
import math as mt
import numpy as np
import pandas as pd
import energyflow as ef

# Function that calculates angles

def pick_angle(y, phi):
    r = mt.sqrt(y**2 + phi**2)
    sin = phi/r
    cos = y/r
    tan = phi/y
    if sin >= 0 and cos >= 0 :
        #print('1st quarter')
        return mt.acos(cos)
    elif sin >= 0 and cos <= 0 :
        #print('2nd quarter')
        return mt.acos(cos)
    elif sin <= 0 and cos <= 0 :
        #print('3rd quarter')
        return -mt.acos(cos)
    elif sin <= 0 and cos >= 0 :
        #print('4th quarter')
        return -mt.acos(cos)


# Takes as arguments 2 vectors, returns dot scalar product

def dot_angle(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    return np.arccos(dot_product)

# Dataframe creator 
# Pass jets_i, jets_i_cols, jets_f, jets_f_cols 
# Returns jets_i + original_line_index + jets_f df, multijetdf

def df_creator(dset, kfactor=None):
    # Create the Dataframe
    i_df = pd.DataFrame(dset.jets_i, columns=dset.jets_i_cols)

    # Create a sample of index positions [0, 1, 2, 3, 4, .....,N-1 ,N]
    col0 = np.linspace(0, len(dset.jets_i), num=len(dset.jets_i), dtype=int, endpoint=False)

    # Insert those indexes into the Dataframe
    # Indexes are inserted after the final column of the original df Dataframe
    end = len(dset.jets_i[0])
    i_df.insert(end, 'index_pos', col0)

    # Create a Dataframe for jets_f array
    f_df = pd.DataFrame(dset.jets_f, columns=dset.jets_f_cols)

    # Concatenate the 2 Dataframes
    df = pd.concat([i_df, f_df], axis=1)

    # For sim/gen the k-factors as last col
    if kfactor=='sim':
    
        factors = ef.mod.kfactors('sim', dset.corr_jet_pts, dset.npvs)
        df = pd.concat([df, pd.DataFrame(factors, columns=['kfactor'])], axis=1)
        m = df['kfactor']*df['weight']
        df = pd.concat([df, pd.DataFrame(m, columns=['weight*kfac'])], axis=1)

    elif kfactor=='gen':

        factors = ef.mod.kfactors('gen', dset.corr_jet_pts)
        df = pd.concat([df, pd.DataFrame(factors, columns=['kfactor'])], axis=1)
        m = df['kfactor']*df['weight']
        df = pd.concat([df, pd.DataFrame(m, columns=['weight*kfac'])], axis=1)

    # Find duplicate rows by 'run number', 'event number' and 'lumi-block number'. Then store them consequtively.
    # For each event, jets sorted by pt-descending order.
    key = df.duplicated(subset=['rn', 'lbn', 'evn'], keep=False)
    multijetdf = df[key]
    multijetdf = multijetdf.sort_values(by=['rn', 'lbn', 'evn','jet_pt'], axis=0, ascending=[True, True, True, False], ignore_index=True)

    return df, multijetdf



#Pass 2 arrays that include as collumns pt|y|phi|mass.
#Jet_array need to be [1, 4] and pfcs [N, :].
#Returns 1d array [y-component, phi-component] of the pull vector.

def pull_vector(jet_array, pfcs_array):
    particle_pt = pfcs_array[:, 0]
    particle_y = pfcs_array[:,1]
    particle_phi = pfcs_array[:,2]

    jet_pt = jet_array[0]
    jet_y = jet_array[1]
    jet_phi = jet_array[2]
    pull_magn = 0
    plv = np.array([0., 0.])

    for i in range(len(particle_pt)):
        ratio = particle_pt[i] / jet_pt
        dy = particle_y[i] - jet_y
        dphi = particle_phi[i] - jet_phi
        pull_magn = ratio*((dy**2 + dphi**2)**(0.5))
        plv[0] += pull_magn*dy
        plv[1] += pull_magn*dphi


    return plv

# A function to calculate relative pull angle and pull angle
# Pass cms, sim or gen_df, their multijet_df counterparts and cms.pfcs, sim.pfcs or gen.gens
# Returns 1d array of relative pull angle and pull angle
# Additionally, weigths, number of pfcs, jet position in multijetdf and info in muons in jets are returned

def rpa_pa_calc(df, multijetdf, pfcs, filt_args = None):

    # Loop elements 
    strt = 0
    fin = len(multijetdf)
    stp = 2
    
    # Column indexes
    n1 = df.columns.get_loc('jet_pt')
    n2 = df.columns.get_loc('jet_y')
    n3 = df.columns.get_loc('jet_phi')

    # Create list to store pull angles
    rel_angle_l = []
    angle_l = []
    weights = []
    num_of_pfcs_jet = []
    jet_orig_loc = []
    muon = []

    # Start the loop

    for line in range(strt, fin, stp):

        # First get the jets and make the vectors
        jet1 = np.asarray(multijetdf.iloc[line])
        jet2 = np.asarray(multijetdf.iloc[line + 1])

        jet1_pt = jet1[n1]
        jet1_y = jet1[n2]
        jet1_phi = jet1[n3]

        jet2_pt = jet2[n1]
        jet2_y = jet2[n2]
        jet2_phi = jet2[n3]

        # Then pfcs for each jet and vectors
        # Jet1 pfcs
        i1 = multijetdf['index_pos'].iloc[line]
        pfcs_jet1 = pfcs[i1]

        # Jet2 pfcs
        i2 = multijetdf['index_pos'].iloc[line + 1]
        pfcs_jet2 = pfcs[i2]

        # Filtering
        if filt_args==None:
            filt_args = {'which':'all', 'pt_cut':0}

        pfcs_jet1 = pfcs_jet1[ef.mod.filter_particles(pfcs_jet1, **filt_args)]
        pfcs_jet2 = pfcs_jet2[ef.mod.filter_particles(pfcs_jet2, **filt_args)]
        
        if len(pfcs_jet1)<8 or len(pfcs_jet2)<8:
            continue

        num_of_pfcs_jet.append(len(pfcs_jet1))
        num_of_pfcs_jet.append(len(pfcs_jet2))

        # Pull vector 
        pull1 = pull_vector(jet1[n1:n3+2], pfcs_jet1)
        pull2 = pull_vector(jet2[n1:n3+2], pfcs_jet2)

        # Connection vector
        connection21 = np.array([jet2_y - jet1_y, jet2_phi - jet1_phi])
        connection12 = np.array([jet1_y - jet2_y, jet1_phi - jet2_phi])

        # Relative Pull angle
        angle21 = dot_angle(pull1, connection21)
        angle12 = dot_angle(pull2, connection12)

        # Pull angle 
        pla21 = pick_angle(pull1[0], pull1[1])
        pla12 = pick_angle(pull2[0], pull2[1])

        # An array to store the relative pull angles
        rel_angle_l.append(angle21)
        rel_angle_l.append(angle12)

        # An array to store the pull angles
        angle_l.append(pla21)
        angle_l.append(pla12)

        # An array to store the weights
        weights.append(multijetdf.iloc[line, -1])
        weights.append(multijetdf.iloc[line+1, -1])

        # An array to store position of jet in multijetdf
        jet_orig_loc.append(line)
        jet_orig_loc.append(line+1)
        
        # A list to store if the jet contains a muon or antimuon
        x1 = (abs(pfcs_jet1[:, -2])==13).any()
        muon.append(x1)
        x1 = (abs(pfcs_jet2[:, -2])==13).any()
        muon.append(x1)

    return np.asarray(rel_angle_l), np.asarray(angle_l), np.asarray(weights), np.asarray(num_of_pfcs_jet), np.asarray(jet_orig_loc), np.asarray(muon) 


# A function for handling boundary conditions
def boundary_cond(p):

    while p > mt.pi:
        p -= 2*mt.pi
    while p < -mt.pi:
        p += 2*mt.pi
    
    return abs(p)

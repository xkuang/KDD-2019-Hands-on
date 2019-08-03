import numpy as np

import torch
from torch.utils.data import Dataset
import dgl

# TrackML Dataset and collate

def get_edge_indices(edges):
    edge_pairs = []
    for i, neighbors in enumerate(edges):
        for e_idx in neighbors:
            edge_pairs.append([i,e_idx])
    return edge_pairs

def get_true_edge_values(pred_edge_idx, true_edges):
    values = [0] * len(pred_edge_idx)
    for i, (src, dst) in enumerate(pred_edge_idx):
        if dst in true_edges[src]:
            values[i] = 1
    return values

class TrackML_Dataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, index):
        s = self.samples[index]
        
        hits = s['hits']
        xyz  = hits['xyz']
        emb  = hits['emb']
        pid  = torch.FloatTensor(hits['particle_id'])
        weight  = torch.FloatTensor(hits['weight'])
        hits = torch.FloatTensor(np.concatenate((xyz, emb), axis=1))
        

        graphs = s['graphs']
        pred_edges = graphs['pred']
        loss_edges = graphs['loss']
        true_edges = graphs['true']
    
        pred_edge_idx = get_edge_indices(pred_edges)
        true_edge_idx = get_edge_indices(loss_edges)
        true_edge_values = get_true_edge_values(true_edge_idx,true_edges)

        # Build inference graph
        g_input = dgl.DGLGraph()
        g_input.add_nodes(len(hits))
        src, dst = tuple(zip(*pred_edge_idx))
        g_input.add_edges(src, dst)
        g_input.ndata['feat'] = hits
        g_input.ndata['pid'] = pid
        g_input.ndata['weight'] = weight

        # Build ground truth graph
        g_true = dgl.DGLGraph()
        g_true.add_nodes(len(hits))
        src, dst = tuple(zip(*true_edge_idx))
        g_true.add_edges(src, dst)
        g_true.edata['truth'] = torch.FloatTensor(true_edge_values)
        
        g_input.set_n_initializer(dgl.init.zero_initializer)
        g_true.set_n_initializer(dgl.init.zero_initializer)
        g_input.set_e_initializer(dgl.init.zero_initializer)
        g_true.set_e_initializer(dgl.init.zero_initializer)
        
        return g_input, g_true
    
    def __len__(self):
        return len(self.samples)
    
def trackml_collate(sample):
    g_input = [s[0] for s in sample]
    g_input = dgl.batch(g_input)

    g_true = [s[1] for s in sample]
    g_true = dgl.batch(g_true)

    return g_input, g_true




"""
TrackML scoring metric (by Sabrina Amrouche, David Rousseau, Moritz Kiehn, Ilija Vukotic)
"""

import pandas

def _analyze_tracks(truth, submission):
    particles_nhits = truth['particle_id'].value_counts(sort=False)
    total_weight = truth['weight'].sum()
    event = pandas.merge(truth[['hit_id', 'particle_id', 'weight']],
                         submission[['hit_id', 'track_id']],
                         on=['hit_id'], how='left', validate='one_to_one')
    event.drop('hit_id', axis=1, inplace=True)
    event.sort_values(by=['track_id', 'particle_id'], inplace=True)


    tracks = []
    rec_track_id = -1
    rec_nhits = 0
    cur_particle_id = -1
    cur_nhits = 0
    cur_weight = 0
    maj_particle_id = -1
    maj_nhits = 0
    maj_weight = 0

    for hit in event.itertuples(index=False):
        if (rec_track_id != -1) and (rec_track_id != hit.track_id):
            if maj_nhits < cur_nhits:
                maj_particle_id = cur_particle_id
                maj_nhits = cur_nhits
                maj_weight = cur_weight
            tracks.append((rec_track_id, rec_nhits, maj_particle_id,
                particles_nhits[maj_particle_id], maj_nhits,
                maj_weight / total_weight))

        if rec_track_id != hit.track_id:
            rec_track_id = hit.track_id
            rec_nhits = 1
            cur_particle_id = hit.particle_id
            cur_nhits = 1
            cur_weight = hit.weight
            maj_particle_id = -1
            maj_nhits = 0
            maj_weights = 0
            continue

        rec_nhits += 1

        if cur_particle_id != hit.particle_id:
            if maj_nhits < cur_nhits:
                maj_particle_id = cur_particle_id
                maj_nhits = cur_nhits
                maj_weight = cur_weight
            cur_particle_id = hit.particle_id
            cur_nhits = 1
            cur_weight = hit.weight
        else:
            cur_nhits += 1
            cur_weight += hit.weight

    if maj_nhits < cur_nhits:
        maj_particle_id = cur_particle_id
        maj_nhits = cur_nhits
        maj_weight = cur_weight
    tracks.append((rec_track_id, rec_nhits, maj_particle_id,
        particles_nhits[maj_particle_id], maj_nhits, maj_weight / total_weight))

    cols = ['track_id', 'nhits',
            'major_particle_id', 'major_particle_nhits',
            'major_nhits', 'major_weight']
    return pandas.DataFrame.from_records(tracks, columns=cols)

def score_event(truth, submission):
    tracks = _analyze_tracks(truth, submission)
    purity_rec = np.true_divide(tracks['major_nhits'], tracks['nhits'])
    purity_maj = np.true_divide(tracks['major_nhits'], tracks['major_particle_nhits'])
    good_track = (0.5 < purity_rec) & (0.5 < purity_maj)
    return tracks['major_weight'][good_track].sum()
"""
Data Generator for fat128 topology
==================================
Converts DatanetAPI samples into the ragged tensor format
required by RouteNetFermiPyTorch.
"""

import os
import sys
import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from fat_tree.datanetAPI import DatanetAPI
import networkx as nx

POLICIES = np.array(['WFQ', 'SP', 'DRR', 'FIFO'])


def network_to_hypergraph(G, R, T, P):
    D_G = nx.DiGraph()
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:
                if G.has_edge(src, dst):
                    D_G.add_node(
                        'l_{}_{}'.format(src, dst),
                        capacity=G.edges[src, dst]['bandwidth'],
                        policy=np.where(G.nodes[src]['schedulingPolicy'] == POLICIES)[0][0]
                    )
                for f_id in range(len(T[src, dst]['Flows'])):
                    if T[src, dst]['Flows'][f_id]['AvgBw'] == 0 or T[src, dst]['Flows'][f_id]['PktsGen'] == 0:
                        continue

                    time_dist_params = [0] * 8
                    flow = T[src, dst]['Flows'][f_id]
                    model = flow['TimeDist'].value
                    if model == 6 and flow['TimeDistParams']['Distribution'] == 'AR1-1':
                        model += 1
                    for param, key in [(0, 'EqLambda'), (1, 'AvgPktsLambda'), (2, 'ExpMaxFactor'),
                                       (3, 'PktsLambdaOn'), (4, 'AvgTOff'), (5, 'AvgTOn'),
                                       (6, 'AR-a'), (7, 'sigma')]:
                        if key in flow['TimeDistParams']:
                            time_dist_params[param] = flow['TimeDistParams'][key]

                    D_G.add_node(
                        'p_{}_{}_{}'.format(src, dst, f_id),
                        source=src, destination=dst,
                        tos=int(T[src, dst]['Flows'][0]['ToS']),
                        traffic=T[src, dst]['Flows'][f_id]['AvgBw'],
                        packets=T[src, dst]['Flows'][f_id]['PktsGen'],
                        length=len(R[src, dst]) - 1,
                        model=model,
                        eq_lambda=time_dist_params[0],
                        avg_pkts_lambda=time_dist_params[1],
                        exp_max_factor=time_dist_params[2],
                        pkts_lambda_on=time_dist_params[3],
                        avg_t_off=time_dist_params[4],
                        avg_t_on=time_dist_params[5],
                        ar_a=time_dist_params[6],
                        sigma=time_dist_params[7],
                        delay=P[src, dst]['Flows'][f_id]['AvgDelay']
                    )

                    for h_1, h_2 in [R[src, dst][i:i+2] for i in range(0, len(R[src, dst]) - 1)]:
                        D_G.add_edge('l_{}_{}'.format(h_1, h_2), 'p_{}_{}_{}'.format(src, dst, f_id))
                        D_G.add_edge('p_{}_{}_{}'.format(src, dst, f_id), 'l_{}_{}'.format(h_1, h_2))

                        q_s = str(G.nodes[h_1]['bufferSizes']).split(',')
                        if 'schedulingWeights' in G.nodes[h_1]:
                            if G.nodes[h_1]['schedulingWeights'] != '-':
                                q_w = [float(w) for w in str(G.nodes[h_1]['schedulingWeights']).split(',')]
                                q_w = [w / sum(q_w) for w in q_w]
                            else:
                                q_w = ['-']
                        else:
                            q_w = ['-']
                        if 'tosToQoSqueue' in G.nodes[h_1]:
                            q_map = [m.split(',') for m in str(G.nodes[h_1]['tosToQoSqueue']).split(';')]
                        else:
                            q_map = [['0'], ['1'], ['2']]
                        q_n = 0
                        if 'levelsQoS' not in G.nodes[h_1]:
                            G.nodes[h_1]['levelsQoS'] = 1
                        for q in range(G.nodes[h_1]['levelsQoS']):
                            D_G.add_node(
                                'q_{}_{}_{}'.format(h_1, h_2, q),
                                queue_size=int(q_s[q]),
                                priority=q_n,
                                weight=q_w[q] if q_w[0] != '-' else 0
                            )
                            D_G.add_edge('q_{}_{}_{}'.format(h_1, h_2, q), 'l_{}_{}'.format(h_1, h_2))
                            if str(int(T[src, dst]['Flows'][0]['ToS'])) in q_map[q]:
                                D_G.add_edge('p_{}_{}_{}'.format(src, dst, f_id), 'q_{}_{}_{}'.format(h_1, h_2, q))
                                D_G.add_edge('q_{}_{}_{}'.format(h_1, h_2, q), 'p_{}_{}_{}'.format(src, dst, f_id))
                            q_n += 1

    D_G.remove_nodes_from([n for n, d in D_G.in_degree() if d == 0])
    return D_G


def hypergraph_to_input_data(HG):
    n_q, n_p, n_l = 0, 0, 0
    mapping = {}
    for entity in list(HG.nodes()):
        if entity.startswith('q'):
            mapping[entity] = 'q_{}'.format(n_q); n_q += 1
        elif entity.startswith('p'):
            mapping[entity] = 'p_{}'.format(n_p); n_p += 1
        elif entity.startswith('l'):
            mapping[entity] = 'l_{}'.format(n_l); n_l += 1
    HG = nx.relabel_nodes(HG, mapping)

    link_to_path, queue_to_path, path_to_queue = [], [], []
    queue_to_link, path_to_link = [], []

    for node in HG.nodes:
        in_nodes = [s for s, d in HG.in_edges(node)]
        if node.startswith('q_'):
            path = []
            for n in in_nodes:
                if n.startswith('p_'):
                    path_pos = [d for _, d in HG.out_edges(n) if d.startswith('q_')]
                    path.append([int(n.replace('p_', '')), path_pos.index(node) if node in path_pos else 0])
            path_to_queue.append(path)
        elif node.startswith('p_'):
            links, queues = [], []
            for n in in_nodes:
                if n.startswith('l_'): links.append(int(n.replace('l_', '')))
                elif n.startswith('q_'): queues.append(int(n.replace('q_', '')))
            link_to_path.append(links)
            queue_to_path.append(queues)
        elif node.startswith('l_'):
            queues, paths = [], []
            for n in in_nodes:
                if n.startswith('q_'): queues.append(int(n.replace('q_', '')))
                elif n.startswith('p_'):
                    path_pos = [d for _, d in HG.out_edges(n) if d.startswith('l_')]
                    paths.append([int(n.replace('p_', '')), path_pos.index(node) if node in path_pos else 0])
            path_to_link.append(paths)
            queue_to_link.append(queues)

    def get_vals(attr, key):
        v = attr.get(key, {})
        return list(v.values()) if v else []

    attrs = nx.get_node_attributes
    return {
        'traffic':          np.expand_dims(get_vals(attrs(HG, 'traffic'),          'traffic'), 1),
        'packets':          np.expand_dims(get_vals(attrs(HG, 'packets'),          'packets'), 1),
        'length':           get_vals(attrs(HG, 'length'),           'length'),
        'model':            get_vals(attrs(HG, 'model'),            'model'),
        'eq_lambda':        np.expand_dims(get_vals(attrs(HG, 'eq_lambda'),        'eq_lambda'), 1),
        'avg_pkts_lambda':  np.expand_dims(get_vals(attrs(HG, 'avg_pkts_lambda'),  'avg_pkts_lambda'), 1),
        'exp_max_factor':   np.expand_dims(get_vals(attrs(HG, 'exp_max_factor'),   'exp_max_factor'), 1),
        'pkts_lambda_on':   np.expand_dims(get_vals(attrs(HG, 'pkts_lambda_on'),   'pkts_lambda_on'), 1),
        'avg_t_off':        np.expand_dims(get_vals(attrs(HG, 'avg_t_off'),        'avg_t_off'), 1),
        'avg_t_on':         np.expand_dims(get_vals(attrs(HG, 'avg_t_on'),         'avg_t_on'), 1),
        'ar_a':             np.expand_dims(get_vals(attrs(HG, 'ar_a'),             'ar_a'), 1),
        'sigma':            np.expand_dims(get_vals(attrs(HG, 'sigma'),            'sigma'), 1),
        'capacity':         np.expand_dims(get_vals(attrs(HG, 'capacity'),         'capacity'), 1),
        'queue_size':       np.expand_dims(get_vals(attrs(HG, 'queue_size'),       'queue_size'), 1),
        'policy':           get_vals(attrs(HG, 'policy'),           'policy'),
        'priority':         get_vals(attrs(HG, 'priority'),         'priority'),
        'weight':           np.expand_dims(get_vals(attrs(HG, 'weight'),           'weight'), 1),
        'delay':            get_vals(attrs(HG, 'delay'),            'delay'),
        'link_to_path':     link_to_path,
        'queue_to_path':    queue_to_path,
        'queue_to_link':    queue_to_link,
        'path_to_queue':    path_to_queue,
        'path_to_link':     path_to_link,
    }

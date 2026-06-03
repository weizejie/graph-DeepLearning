# -*- coding: utf-8 -*-
"""Cross-validation: TF model vs converted PyTorch model."""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import torch
import tensorflow as tf
import networkx as nx

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.insert(0, os.path.join(project_root, 'fat_tree'))
from delay_model import RouteNet_Fermi as TFModel
from data_generator import input_fn, hypergraph_to_input_data as tf_hypergraph_to_input_data, network_to_hypergraph as tf_network_to_hypergraph

sys.path.insert(0, project_root)
from topology_transfer.routenet_fermi_pytorch import RouteNetFermiPyTorch

POLICIES = np.array(['WFQ', 'SP', 'DRR', 'FIFO'])


def network_to_hypergraph_pytorch(G, R, T, P):
    """PyTorch version: same logic as TF data_generator.py."""
    D_G = nx.DiGraph()
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:
                if G.has_edge(src, dst):
                    policy_str = G.nodes[src]['schedulingPolicy'] if 'schedulingPolicy' in G.nodes[src] else 'FIFO'
                    policy_val = np.where(policy_str == POLICIES)[0]
                    policy_code = policy_val[0] if len(policy_val) > 0 else 3
                    D_G.add_node(
                        'l_{}_{}'.format(src, dst),
                        capacity=G.edges[src, dst]['bandwidth'],
                        policy=policy_code
                    )
                for f_id in range(len(T[src, dst]['Flows'])):
                    if T[src, dst]['Flows'][f_id]['AvgBw'] != 0 and T[src, dst]['Flows'][f_id]['PktsGen'] != 0:
                        time_dist_params = [0] * 8
                        flow = T[src, dst]['Flows'][f_id]
                        model = flow['TimeDist'].value
                        # Clone TimeDistParams to avoid mutating the shared dict
                        tdp = dict(flow['TimeDistParams'])
                        if model == 6 and tdp.get('Distribution') == 'AR1-1':
                            model += 1
                        tdp['Distribution'] = 'Poisson'
                        if 'EqLambda' in tdp:
                            time_dist_params[0] = tdp['EqLambda']
                        if 'AvgPktsLambda' in tdp:
                            time_dist_params[1] = tdp['AvgPktsLambda']
                        if 'ExpMaxFactor' in tdp:
                            time_dist_params[2] = tdp['ExpMaxFactor']
                        if 'PktsLambdaOn' in tdp:
                            time_dist_params[3] = tdp['PktsLambdaOn']
                        if 'AvgTOff' in tdp:
                            time_dist_params[4] = tdp['AvgTOff']
                        if 'AvgTOn' in tdp:
                            time_dist_params[5] = tdp['AvgTOn']
                        if 'AR-a' in tdp:
                            time_dist_params[6] = tdp['AR-a']
                        if 'sigma' in tdp:
                            time_dist_params[7] = tdp['sigma']
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
                            levelsQoS = G.nodes[h_1].get('levelsQoS', 1)
                            for q in range(levelsQoS):
                                D_G.add_node(
                                    'q_{}_{}_{}'.format(h_1, h_2, q),
                                    queue_size=int(q_s[q]) if q < len(q_s) else 1000,
                                    priority=q_n,
                                    weight=q_w[q] if q_w[0] != '-' else 0
                                )
                                D_G.add_edge('q_{}_{}_{}'.format(h_1, h_2, q), 'l_{}_{}'.format(h_1, h_2))
                                if str(int(T[src, dst]['Flows'][0]['ToS'])) in q_map[q]:
                                    D_G.add_edge('p_{}_{}_{}'.format(src, dst, f_id), 'q_{}_{}_{}'.format(h_1, h_2, q))
                                    D_G.add_edge('q_{}_{}_{}'.format(h_1, h_2, q), 'p_{}_{}_{}'.format(src, dst, f_id))
                                q_n += 1
    D_G.remove_nodes_from([node for node, in_degree in D_G.in_degree() if in_degree == 0])
    return D_G


def hypergraph_to_input_data_pytorch(HG):
    """PyTorch version: same logic as TF data_generator.py."""
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

    def safe_vals(attr_dict):
        """Extract values from a {node_name: value} dict."""
        if isinstance(attr_dict, dict):
            return list(attr_dict.values())
        return []

    result = {
        'traffic': np.expand_dims(safe_vals(nx.get_node_attributes(HG, 'traffic')), 1),
        'packets': np.expand_dims(safe_vals(nx.get_node_attributes(HG, 'packets')), 1),
        'length': safe_vals(nx.get_node_attributes(HG, 'length')),
        'model': safe_vals(nx.get_node_attributes(HG, 'model')),
        'eq_lambda': np.expand_dims(safe_vals(nx.get_node_attributes(HG, 'eq_lambda')), 1),
        'avg_pkts_lambda': np.expand_dims(safe_vals(nx.get_node_attributes(HG, 'avg_pkts_lambda')), 1),
        'exp_max_factor': np.expand_dims(safe_vals(nx.get_node_attributes(HG, 'exp_max_factor')), 1),
        'pkts_lambda_on': np.expand_dims(safe_vals(nx.get_node_attributes(HG, 'pkts_lambda_on')), 1),
        'avg_t_off': np.expand_dims(safe_vals(nx.get_node_attributes(HG, 'avg_t_off')), 1),
        'avg_t_on': np.expand_dims(safe_vals(nx.get_node_attributes(HG, 'avg_t_on')), 1),
        'ar_a': np.expand_dims(safe_vals(nx.get_node_attributes(HG, 'ar_a')), 1),
        'sigma': np.expand_dims(safe_vals(nx.get_node_attributes(HG, 'sigma')), 1),
        'capacity': np.expand_dims(safe_vals(nx.get_node_attributes(HG, 'capacity')), 1),
        'queue_size': np.expand_dims(safe_vals(nx.get_node_attributes(HG, 'queue_size')), 1),
        'policy': safe_vals(nx.get_node_attributes(HG, 'policy')),
        'priority': safe_vals(nx.get_node_attributes(HG, 'priority')),
        'weight': np.expand_dims(safe_vals(nx.get_node_attributes(HG, 'weight')), 1),
        'delay': safe_vals(nx.get_node_attributes(HG, 'delay')),
        'link_to_path': link_to_path,
        'queue_to_path': queue_to_path,
        'queue_to_link': queue_to_link,
        'path_to_queue': path_to_queue,
        'path_to_link': path_to_link,
    }

    return result


def prepare_pytorch_inputs(data, device='cpu'):
    """Convert dict to PyTorch model input."""
    num_paths = len(data['traffic'])
    num_links = len(data['capacity'])
    num_queues = len(data['queue_size'])

    return {
        'traffic': torch.tensor(data['traffic'], dtype=torch.float32, device=device),
        'packets': torch.tensor(data['packets'], dtype=torch.float32, device=device),
        'length': torch.tensor(data['length'], dtype=torch.long, device=device),
        'model': torch.tensor(data['model'], dtype=torch.long, device=device),
        'eq_lambda': torch.tensor(data['eq_lambda'], dtype=torch.float32, device=device),
        'avg_pkts_lambda': torch.tensor(data['avg_pkts_lambda'], dtype=torch.float32, device=device),
        'exp_max_factor': torch.tensor(data['exp_max_factor'], dtype=torch.float32, device=device),
        'pkts_lambda_on': torch.tensor(data['pkts_lambda_on'], dtype=torch.float32, device=device),
        'avg_t_off': torch.tensor(data['avg_t_off'], dtype=torch.float32, device=device),
        'avg_t_on': torch.tensor(data['avg_t_on'], dtype=torch.float32, device=device),
        'ar_a': torch.tensor(data['ar_a'], dtype=torch.float32, device=device),
        'sigma': torch.tensor(data['sigma'], dtype=torch.float32, device=device),
        'capacity': torch.tensor(data['capacity'], dtype=torch.float32, device=device),
        'policy': torch.tensor(data['policy'], dtype=torch.long, device=device),
        'queue_size': torch.tensor(data['queue_size'], dtype=torch.float32, device=device),
        'priority': torch.tensor(data['priority'], dtype=torch.long, device=device),
        'weight': torch.tensor(data['weight'], dtype=torch.float32, device=device),
        'link_to_path': data['link_to_path'],
        'queue_to_path': data['queue_to_path'],
        'path_to_link': data['path_to_link'],
        'path_to_queue': data['path_to_queue'],
        'queue_to_link': data['queue_to_link'],
        'num_paths': num_paths,
        'num_queues': num_queues,
        'num_links': num_links,
    }


def run_tf_model(data, tf_ckpt_path):
    """Run TF model and return predictions using predict()."""
    print("  [TF] Loading model and checkpoint...")
    tf_model = TFModel()
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    loss_object = tf.keras.losses.MeanAbsolutePercentageError()
    tf_model.compile(loss=loss_object, optimizer=optimizer)
    tf_model.load_weights(tf_ckpt_path)

    # Build a single-sample tf.data.Dataset to match expected input signature
    tf_input = {
        'traffic': tf.constant(data['traffic'], dtype=tf.float32),
        'packets': tf.constant(data['packets'], dtype=tf.float32),
        'length': tf.constant(data['length'], dtype=tf.int32),
        'model': tf.constant(data['model'], dtype=tf.int32),
        'eq_lambda': tf.constant(data['eq_lambda'], dtype=tf.float32),
        'avg_pkts_lambda': tf.constant(data['avg_pkts_lambda'], dtype=tf.float32),
        'exp_max_factor': tf.constant(data['exp_max_factor'], dtype=tf.float32),
        'pkts_lambda_on': tf.constant(data['pkts_lambda_on'], dtype=tf.float32),
        'avg_t_off': tf.constant(data['avg_t_off'], dtype=tf.float32),
        'avg_t_on': tf.constant(data['avg_t_on'], dtype=tf.float32),
        'ar_a': tf.constant(data['ar_a'], dtype=tf.float32),
        'sigma': tf.constant(data['sigma'], dtype=tf.float32),
        'capacity': tf.constant(data['capacity'], dtype=tf.float32),
        'queue_size': tf.constant(data['queue_size'], dtype=tf.float32),
        'policy': tf.constant(data['policy'], dtype=tf.int32),
        'priority': tf.constant(data['priority'], dtype=tf.int32),
        'weight': tf.constant(data['weight'], dtype=tf.float32),
        'link_to_path': data['link_to_path'],      # already tf.RaggedTensorValue
        'queue_to_path': data['queue_to_path'],   # already tf.RaggedTensorValue
        'queue_to_link': data['queue_to_link'],   # already tf.RaggedTensorValue
        'path_to_queue': data['path_to_queue'],   # already tf.RaggedTensorValue
        'path_to_link': data['path_to_link'],     # already tf.RaggedTensorValue
    }

    print("  [TF] Running forward pass...")
    ds = tf.data.Dataset.from_tensors(tf_input)
    predictions = tf_model.predict(ds, verbose=0)
    return predictions.flatten()


def run_pytorch_model(data, pt_ckpt_path, device='cpu'):
    """Run PyTorch model and return predictions."""
    print("  [PT] Loading model and checkpoint...")
    pt_model = RouteNetFermiPyTorch().to(device)
    ckpt = torch.load(pt_ckpt_path, map_location=device)
    pt_model.load_state_dict(ckpt['model_state_dict'])
    pt_model.eval()

    inputs = prepare_pytorch_inputs(data, device=device)
    print("  [PT] Running forward pass...")
    with torch.no_grad():
        result = pt_model(inputs)
    return result.cpu().numpy()


def verify(data_dir, tf_ckpt_path, pt_ckpt_path, sample_idx=0, device='cpu'):
    """Load sample, run both models, compare outputs."""
    from datanetAPI import DatanetAPI

    print("\n" + "=" * 60)
    print("Cross-validation: TF vs converted PyTorch")
    print("  Data dir: {}".format(data_dir))
    print("  TF checkpoint: {}".format(tf_ckpt_path))
    print("  PT checkpoint: {}".format(pt_ckpt_path))
    print("  Sample index: {}".format(sample_idx))
    print("=" * 60 + "\n")

    print("[Setup] Loading sample {} via DatanetAPI...".format(sample_idx))
    tool = DatanetAPI(data_dir, shuffle=False)
    for i, sample in enumerate(tool):
        if i < sample_idx:
            continue
        G = nx.DiGraph(sample.get_topology_object())
        T = sample.get_traffic_matrix()
        R = sample.get_routing_matrix()
        P = sample.get_performance_matrix()
        break

    # Build hypergraph with TF loader (gives us both dict and delay)
    HG_tf = tf_network_to_hypergraph(G, R, T, P)
    HG_tf_raw = tf_hypergraph_to_input_data(HG_tf)
    tf_data = dict(HG_tf_raw[0])  # copy

    # Build hypergraph with PyTorch loader
    HG_pt = network_to_hypergraph_pytorch(G, R, T, P)
    pt_data = hypergraph_to_input_data_pytorch(HG_pt)

    print("\n[Data] TF: {} paths, {} links, {} queues".format(
        len(tf_data['traffic']), len(tf_data['capacity']), len(tf_data['queue_size'])))
    print("[Data] PT: {} paths, {} links, {} queues".format(
        len(pt_data['traffic']), len(pt_data['capacity']), len(pt_data['queue_size'])))

    # Check data consistency
    for key in ['traffic', 'packets', 'length', 'model', 'capacity', 'queue_size']:
        tf_arr = np.array(tf_data[key])
        pt_arr = np.array(pt_data[key])
        match = np.allclose(tf_arr, pt_arr, rtol=1e-4, atol=1e-6)
        print("[Data] {:12s}: TF shape={:20s}  PT shape={:20s}  match={}".format(
            key, str(tf_arr.shape), str(pt_arr.shape), match))
        if not match:
            diff = np.abs(tf_arr - pt_arr).max()
            print("        max diff = {}".format(diff))

    # Run both models
    tf_preds = run_tf_model(tf_data, tf_ckpt_path)
    pt_preds = run_pytorch_model(pt_data, pt_ckpt_path, device)

    print("\n[Result] TF: min={:.6f}, max={:.6f}, mean={:.6f}".format(
        tf_preds.min(), tf_preds.max(), tf_preds.mean()))
    print("[Result] PT: min={:.6f}, max={:.6f}, mean={:.6f}".format(
        pt_preds.min(), pt_preds.max(), pt_preds.mean()))

    # Compare
    abs_diff = np.abs(tf_preds - pt_preds)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())

    print("\n" + "=" * 60)
    print("Comparison:")
    print("  Max absolute diff: {:.8f}".format(max_diff))
    print("  Mean absolute diff: {:.8f}".format(mean_diff))
    for thresh in [1e-3, 1e-2, 1e-1, 1.0]:
        cnt = int((abs_diff < thresh).sum())
        print("  Flows with |diff| < {}: {} / {}".format(thresh, cnt, len(abs_diff)))
    print("=" * 60 + "\n")

    return max_diff, mean_diff


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=r'D:\newdownload\RouteNet-Fermi-main (1)\RouteNet-Fermi-main\fat128\test')
    parser.add_argument('--tf_ckpt', type=str,
                        default='../fat_tree/ckpt_dir_128/02-0.59')
    parser.add_argument('--pt_ckpt', type=str,
                        default='./checkpoints/converted_from_tf.pt')
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    max_diff, mean_diff = verify(
        args.data_dir, args.tf_ckpt, args.pt_ckpt,
        sample_idx=args.sample_idx, device=args.device
    )

    if max_diff < 1e-3:
        print("PASS: Outputs match within tolerance (max_diff < 1e-3)")
        print("The TF -> PyTorch conversion is VERIFIED.")
        sys.exit(0)
    else:
        print("FAIL: Outputs differ beyond tolerance (max_diff >= 1e-3)")
        print("The conversion has issues -- see above for details.")
        sys.exit(1)

"""
Training script for RouteNet-Fermi on fat128 topology (PyTorch)
==============================================================
Trains the model on fat128 data, saves checkpoints.
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from topology_transfer.routenet_fermi_pytorch import RouteNetFermiPyTorch
from fat_tree.datanetAPI import DatanetAPI
import networkx as nx

POLICIES = np.array(['WFQ', 'SP', 'DRR', 'FIFO'])


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    attrs = nx.get_node_attributes
    return {
        'traffic':          np.expand_dims(list(attrs(HG, 'traffic').values()),          1),
        'packets':          np.expand_dims(list(attrs(HG, 'packets').values()),          1),
        'length':           list(attrs(HG, 'length').values()),
        'model':            list(attrs(HG, 'model').values()),
        'eq_lambda':        np.expand_dims(list(attrs(HG, 'eq_lambda').values()),        1),
        'avg_pkts_lambda':  np.expand_dims(list(attrs(HG, 'avg_pkts_lambda').values()),  1),
        'exp_max_factor':   np.expand_dims(list(attrs(HG, 'exp_max_factor').values()),   1),
        'pkts_lambda_on':   np.expand_dims(list(attrs(HG, 'pkts_lambda_on').values()),   1),
        'avg_t_off':        np.expand_dims(list(attrs(HG, 'avg_t_off').values()),        1),
        'avg_t_on':         np.expand_dims(list(attrs(HG, 'avg_t_on').values()),         1),
        'ar_a':             np.expand_dims(list(attrs(HG, 'ar_a').values()),             1),
        'sigma':            np.expand_dims(list(attrs(HG, 'sigma').values()),            1),
        'capacity':         np.expand_dims(list(attrs(HG, 'capacity').values()),         1),
        'queue_size':       np.expand_dims(list(attrs(HG, 'queue_size').values()),       1),
        'policy':           list(attrs(HG, 'policy').values()),
        'priority':         list(attrs(HG, 'priority').values()),
        'weight':           np.expand_dims(list(attrs(HG, 'weight').values()),           1),
        'delay':            list(attrs(HG, 'delay').values()),
        'link_to_path':     link_to_path,
        'queue_to_path':    queue_to_path,
        'queue_to_link':    queue_to_link,
        'path_to_queue':    path_to_queue,
        'path_to_link':     path_to_link,
    }


class Fat128Dataset:
    """Memory-efficient lazy dataset — matches TF source generator behavior exactly.
    Holds only an iterator, yielding one sample at a time.
    Each __getitem__ call loads and processes exactly one sample.
    """
    def __init__(self, data_dir, shuffle=False):
        self.data_dir = data_dir
        self.tool = DatanetAPI(data_dir, shuffle=shuffle, seed=1234)
        self._iter = iter(self.tool)
        self._cached = None   # holds last yielded sample for next __getitem__
        self._exhausted = False
        print(f"  Dataset iterator ready: {data_dir}")

    def __len__(self):
        raise NotImplementedError("Lazy dataset has no len() — use steps_per_epoch to limit training steps")

    def __getitem__(self, idx):
        # Persistent iterator — each call yields the next valid sample from DatanetAPI.
        # DatanetAPI internally catches corrupt tar files and skips to the next file,
        # but may yield a partially-loaded Sample. We validate at every stage and
        # retry on bad data, so the training loop always receives clean (data, delay).
        MAX_RETRIES = 10
        for attempt in range(MAX_RETRIES):
            if self._cached is not None:
                sample = self._cached
                self._cached = None
            elif self._exhausted:
                raise StopIteration("Dataset exhausted")
            else:
                try:
                    sample = next(self._iter)
                except StopIteration:
                    self._exhausted = True
                    raise

            try:
                G = nx.DiGraph(sample.get_topology_object())
            except Exception as e:
                continue  # invalid topology → retry

            try:
                T = sample.get_traffic_matrix()
                R = sample.get_routing_matrix()
                P = sample.get_performance_matrix()
            except Exception as e:
                continue  # missing matrices → retry

            # Sanity-check: T must have Flows for at least some src/dst
            if T is None or not isinstance(T, (dict, np.matrix)):
                continue

            try:
                HG = network_to_hypergraph(G=G, R=R, T=T, P=P)
            except Exception as e:
                continue  # hypergraph construction failed → retry

            try:
                data = hypergraph_to_input_data(HG)
            except Exception as e:
                continue  # input data conversion failed → retry

            delay_arr = np.array(data.pop('delay'), dtype=np.float32)
            if len(delay_arr) == 0:
                continue  # empty delay → retry

            return data, delay_arr

        raise RuntimeError(f"Failed to get a valid sample after {MAX_RETRIES} attempts")


def collate_samples(samples, device='cpu'):
    """
    Merge list of (data_dict, delay) into a single batched inputs dict.
    All edge-list indices are rebased into a global coordinate system.
    Per-sample offsets ensure paths/links/queues from different samples occupy
    disjoint ranges in the global index space.

    Output format (matching TF ragged tensor semantics):
      link_to_path:  list[N_paths]  of list[int]          — path → its link IDs
      queue_to_path: list[N_paths]  of list[int]          — path → its queue IDs
      path_to_link:  list[N_links]  of list[(pid,pos)]  — link → paths+pos traversing it
      path_to_queue: list[N_queues] of list[(pid,pos)]  — queue → paths+pos using it
      queue_to_link: list[N_links]  of list[int]          — link → its queue IDs
    """
    data_list, delay_list = zip(*samples)
    n = len(samples)

    # Dense tensors: simple concat
    batched = {}
    for key in ['traffic', 'packets', 'eq_lambda', 'avg_pkts_lambda', 'exp_max_factor',
                'pkts_lambda_on', 'avg_t_off', 'avg_t_on', 'ar_a', 'sigma',
                'capacity', 'queue_size', 'weight']:
        batched[key] = torch.tensor(
            np.concatenate([data_list[i][key] for i in range(n)]),
            dtype=torch.float32
        ).to(device)

    for key, dtype in [('model', torch.long), ('policy', torch.long),
                        ('priority', torch.long), ('length', torch.long)]:
        batched[key] = torch.tensor(
            np.concatenate([data_list[i][key] for i in range(n)]),
            dtype=dtype
        ).to(device)

    batched['delay'] = torch.from_numpy(
        np.concatenate([np.asarray(delay_list[i], dtype=np.float32) for i in range(n)])
    ).clone().to(device)

    num_paths_list  = [len(data_list[i]['traffic'])    for i in range(n)]
    num_links_list  = [len(data_list[i]['capacity'])   for i in range(n)]
    num_queues_list = [len(data_list[i]['queue_size']) for i in range(n)]

    batched['num_paths']  = sum(num_paths_list)
    batched['num_links']  = sum(num_links_list)
    batched['num_queues'] = sum(num_queues_list)

    l2p_out, q2p_out, p2l_out, p2q_out, q2l_out = [], [], [], [], []

    path_off = 0
    link_off = 0
    queue_off = 0

    for i in range(n):
        d = data_list[i]
        np_ = num_paths_list[i]
        nl_ = num_links_list[i]
        nq_ = num_queues_list[i]

        # link_to_path: list[N_paths] of list[link_ids] — rebase link IDs
        for entry in d['link_to_path']:
            l2p_out.append([lid + link_off for lid in entry])

        # queue_to_path: list[N_paths] of list[queue_ids] — rebase queue IDs
        for entry in d['queue_to_path']:
            q2p_out.append([qid + queue_off for qid in entry])

        # queue_to_link: list[N_links] of list[queue_ids] — FIXED direction
        # Original hypergraph_to_input_data already builds this per-link.
        # Append as-is; no rebase needed for queue IDs (use queue_off).
        for entry in d['queue_to_link']:
            q2l_out.append([qid + queue_off for qid in entry])

        # path_to_link: list[N_links] of list[(pid, pos)] — rebase path IDs
        for entry in d['path_to_link']:
            p2l_out.append([[pid + path_off, pos] for pid, pos in entry])

        # path_to_queue: list[N_queues] of list[(pid, pos)] — rebase path IDs
        for entry in d['path_to_queue']:
            p2q_out.append([[pid + path_off, pos] for pid, pos in entry])

        path_off  += np_
        link_off  += nl_
        queue_off += nq_

    batched['link_to_path']  = l2p_out
    batched['queue_to_path'] = q2p_out
    batched['path_to_link']  = p2l_out
    batched['path_to_queue'] = p2q_out
    batched['queue_to_link'] = q2l_out
    batched['path_lens']     = batched['length']

    return batched


def mape_loss(pred, target):
    mask = target > 0
    if not mask.any():
        return torch.tensor(0.0, device=pred.device)
    return (torch.abs((target[mask] - pred[mask]) / target[mask]) * 100).mean()


def mape(pred, target):
    """MAPE metric for logging."""
    return mape_loss(pred, target)



def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f"Device: {device}")

    print(f"Loading training data from: {args.train_dir}")
    train_dataset = Fat128Dataset(args.train_dir, shuffle=True)

    val_dataset = None
    if args.val_dir:
        print(f"Loading validation data from: {args.val_dir}")
        val_dataset = Fat128Dataset(args.val_dir, shuffle=False)

    model = RouteNetFermiPyTorch().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = mape_loss  # same as TF source: MeanAbsolutePercentageError
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_mape = float('inf')

    if args.load_from and os.path.exists(args.load_from):
        print(f"[LOAD] Loading checkpoint from: {args.load_from}")
        print(f"[LOAD] File size: {os.path.getsize(args.load_from):,} bytes")
        ckpt = torch.load(args.load_from, map_location=device, weights_only=False)
        print(f"[LOAD] Checkpoint keys: {list(ckpt.keys())}")
        print(f"[LOAD] State dict params: {len(ckpt['model_state_dict'])}")
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"[LOAD] Missing after load: {missing}")
        print(f"[LOAD] Unexpected after load: {unexpected}")
        val_mape_from_ckpt = ckpt.get('val_mape', None)
        if val_mape_from_ckpt is not None and isinstance(val_mape_from_ckpt, float):
            best_val_mape = val_mape_from_ckpt
            print(f"[LOAD] Done (val_mape={val_mape_from_ckpt:.2f}%), best_val_mape={best_val_mape:.2f}%")
        else:
            print(f"[LOAD] Done (no val_mape in checkpoint)")
    else:
        print("[LOAD] No checkpoint specified or file not found — training from scratch.")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_mape_val = 0.0
        n_batches = 0

        epoch_start = time.time()

        # Direct iteration — matches TF source: for sample in dataset generator
        # Each call to train_dataset[0] yields the next sample from the shuffled iterator
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch", ncols=80)
        batch_samples = []
        for _ in pbar:
            try:
                data, delay = train_dataset[0]
            except StopIteration:
                tqdm.write(f"  Dataset exhausted at step {n_batches}")
                break
            except RuntimeError as e:
                tqdm.write(f"  All retries failed: {e}")
                continue

            batch_samples.append((data, delay))

            if len(batch_samples) < args.batch_size:
                continue

            # Got a full batch — process it
            try:
                inputs = collate_samples(batch_samples, device)
            except Exception as e:
                tqdm.write(f"  Collate error: {e}")
                batch_samples = []
                continue

            batch_samples = []

            optimizer.zero_grad()
            pred = model(inputs)
            # MAPE loss on raw delay — matches TF: loss_object = MeanAbsolutePercentageError()
            loss = mape_loss(pred, inputs['delay'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                # MAPE on raw scale: pred and target are both raw delay
                batch_mape = mape(pred, inputs['delay'])

            train_loss += loss.item()
            train_mape_val += batch_mape.item()
            n_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", mape=f"{batch_mape.item():.2f}%")

            if n_batches % args.log_every == 0:
                print(f"\n  Epoch {epoch+1}/{args.epochs} | Step {n_batches}/{args.steps_per_epoch} | "
                      f"Loss: {loss.item():.4f} | MAPE: {batch_mape.item():.2f}%")

        pbar.close()
        if n_batches == 0:
            print(f"  Warning: no valid batches in epoch {epoch+1}")
            continue

        train_loss /= max(n_batches, 1)
        train_mape_val /= max(n_batches, 1)
        epoch_time = time.time() - epoch_start

        val_mape_val = None
        if val_dataset is not None:
            model.eval()
            val_preds, val_targets = [], []
            val_pbar = tqdm(range(args.val_steps),
                            desc=f"Val {epoch+1}", unit="batch", ncols=60, leave=False)
            with torch.no_grad():
                for _ in val_pbar:
                    batch_samples = []
                    for _ in range(args.batch_size):
                        try:
                            data, delay = val_dataset[0]
                            batch_samples.append((data, delay))
                        except StopIteration:
                            break
                    if not batch_samples:
                        break
                    inputs = collate_samples(batch_samples, device)
                    pred = model(inputs)
                    val_preds.append(pred.cpu())
                    val_targets.append(inputs['delay'].cpu())
            val_pbar.close()
            if val_preds:
                vp = torch.cat(val_preds)
                vt = torch.cat(val_targets)
                # MAPE on raw scale — model outputs raw delay
                val_mape_val = mape(vp, vt).item()

        val_mape_str_ = f"{val_mape_val:.2f}" if val_mape_val is not None else "N/A"
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | "
              f"Train MAPE: {train_mape_val:.2f}% | Val MAPE: {val_mape_val:.2f}% | "
              f"Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save every epoch (matches TF: save_best_only=False, save_freq='epoch')
        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_mape': train_mape_val,
            'val_mape': val_mape_val,
        }
        val_mape_str = f"{val_mape_val:.2f}" if val_mape_val is not None else "N/A"
        ckpt_path = os.path.join(args.output_dir, f'epoch{epoch+1:02d}-{val_mape_str}.pt')
        torch.save(ckpt, ckpt_path)
        torch.save(ckpt, os.path.join(args.output_dir, 'latest.pt'))

        if val_mape_val is not None and (best_val_mape is None or val_mape_val < best_val_mape):
            best_val_mape = val_mape_val
            torch.save(ckpt, os.path.join(args.output_dir, 'best.pt'))
            print(f"  -> New best model saved (val_mape: {val_mape_val:.2f}%)")

    print(f"\nTraining complete. Best val MAPE: {best_val_mape:.2f}%")
    print(f"Models saved to: {args.output_dir}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train RouteNet-Fermi on fat128 (PyTorch)')
    parser.add_argument('--train_dir',   type=str, default='../data/fat128/train')
    parser.add_argument('--val_dir',     type=str, default='../data/fat128/test')
    parser.add_argument('--output_dir',  type=str, default='./checkpoints')
    parser.add_argument('--epochs',          type=int, default=5)
    parser.add_argument('--batch_size',       type=int, default=1)
    parser.add_argument('--lr',              type=float, default=0.001)
    parser.add_argument('--steps_per_epoch',  type=int, default=200)
    parser.add_argument('--log_every',        type=int, default=10)
    parser.add_argument('--val_steps',         type=int, default=50)
    parser.add_argument('--shuffle',          action='store_true')
    parser.add_argument('--seed',             type=int, default=42)
    parser.add_argument('--gpu',              type=int, default=0)
    parser.add_argument('--load_from', type=str,
                        default='./checkpoints/converted_from_tf.pt',
                        \
                        help='Path to .pt checkpoint to load. Set to empty string to train from scratch.')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

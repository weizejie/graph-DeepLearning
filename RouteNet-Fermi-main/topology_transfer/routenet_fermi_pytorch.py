"""
RouteNet-Fermi PyTorch Implementation
====================================
Reimplemented from TensorFlow (delay_model.py) to avoid version conflicts.
Handles graph neural network message passing over paths, links, and queues
with variable-length (ragged) topology structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def to(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list):
        return [to(item, device) for item in x]
    elif isinstance(x, dict):
        return {k: to(v, device) for k, v in x.items()}
    return x


class PathEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class QueueEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class LinkEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class PathReadout(nn.Module):
    def __init__(self, path_state_dim, link_state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(path_state_dim, link_state_dim // 2),
            nn.ReLU(),
            nn.Linear(link_state_dim // 2, path_state_dim // 2),
            nn.ReLU(),
            nn.Linear(path_state_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class RouteNetFermiPyTorch(nn.Module):
    """
    RouteNet-Fermi model for delay prediction in network topologies.
    Uses GNN-style message passing over hypergraphs with:
      - Paths (flows from source to destination)
      - Links (network edges)
      - Queues (scheduling queues at each link)
    """

    def __init__(self):
        super().__init__()

        self.max_num_models = 7
        self.num_policies = 4
        self.max_num_queues = 3
        self.iterations = 8

        self.path_state_dim = 32
        self.link_state_dim = 32
        self.queue_state_dim = 32

        self.z_score = {
            'traffic':          [1385.4058837890625, 859.8118896484375],
            'packets':          [1.4015231132507324, 0.8932565450668335],
            'eq_lambda':         [1350.97119140625, 858.316162109375],
            'avg_pkts_lambda':   [0.9117304086685181, 0.9723503589630127],
            'exp_max_factor':   [6.663637638092041, 4.715115070343018],
            'pkts_lambda_on':    [0.9116322994232178, 1.651275396347046],
            'avg_t_off':        [1.6649284362792969, 2.356407403945923],
            'avg_t_on':         [1.6649284362792969, 2.356407403945923],
            'ar_a':             [0.0, 1.0],
            'sigma':            [0.0, 1.0],
            'capacity':         [27611.091796875, 20090.62109375],
            'queue_size':       [30259.10546875, 21410.095703125],
        }

        self.path_embedding = PathEmbedding(10 + self.max_num_models, self.path_state_dim)
        self.queue_embedding = QueueEmbedding(self.max_num_queues + 2, self.queue_state_dim)
        self.link_embedding = LinkEmbedding(self.num_policies + 1, self.link_state_dim)
        self.readout_path = PathReadout(self.path_state_dim, self.link_state_dim)

        self.path_gru = nn.GRUCell(self.queue_state_dim + self.link_state_dim, self.path_state_dim)
        self.link_gru = nn.GRUCell(self.queue_state_dim, self.link_state_dim)
        self.queue_gru = nn.GRUCell(self.path_state_dim, self.queue_state_dim)

    def _z_norm(self, x, key):
        mu, std = self.z_score[key]
        return (x - mu) / (std + 1e-8)

    def forward(self, inputs):
        """
        Args:
            inputs: dict containing:
                - traffic, packets, eq_lambda, ... : (N, 1) float tensors
                - model: (N,) int tensor
                - capacity: (L, 1) float tensor
                - policy: (L,) int tensor
                - queue_size, weight: (Q, 1) float tensors
                - priority: (Q,) int tensor
                - length / path_lens: (N,) int tensor
                - link_to_path: list of N lists, each with link indices
                - queue_to_path: list of N lists, each with queue indices
                - path_to_link: list of L lists, each [(path_idx, pos), ...]
                - path_to_queue: list of Q lists, each [(path_idx, pos), ...]
                - queue_to_link: list of L lists, each with queue indices
                - num_paths, num_queues, num_links: int

        Returns:
            delay: (N,) predicted delay per path
        """
        device = inputs['traffic'].device

        # ── Extract and validate shapes ─────────────────────────────────────────
        traffic         = inputs['traffic'].squeeze(-1)          # (N,)
        packets         = inputs['packets'].squeeze(-1)          # (N,)
        length          = inputs['length']                       # (N,)
        model           = inputs['model']                        # (N,)
        eq_lambda       = inputs['eq_lambda'].squeeze(-1)        # (N,)
        avg_pkts_lambda = inputs['avg_pkts_lambda'].squeeze(-1)  # (N,)
        exp_max_factor  = inputs['exp_max_factor'].squeeze(-1)   # (N,)
        pkts_lambda_on  = inputs['pkts_lambda_on'].squeeze(-1)   # (N,)
        avg_t_off       = inputs['avg_t_off'].squeeze(-1)        # (N,)
        avg_t_on        = inputs['avg_t_on'].squeeze(-1)         # (N,)
        ar_a            = inputs['ar_a'].squeeze(-1)            # (N,)
        sigma           = inputs['sigma'].squeeze(-1)            # (N,)
        capacity        = inputs['capacity'].squeeze(-1)         # (L,)
        policy          = inputs['policy']                        # (L,)
        queue_size      = inputs['queue_size'].squeeze(-1)       # (Q,)
        priority        = inputs['priority']                      # (Q,)
        weight          = inputs['weight'].squeeze(-1)            # (Q,)
        link_to_path    = inputs['link_to_path']                  # list[N]
        queue_to_path   = inputs['queue_to_path']                 # list[N]
        path_to_link    = inputs['path_to_link']                  # list[L]
        path_to_queue   = inputs['path_to_queue']                # list[Q]
        queue_to_link   = inputs['queue_to_link']                # list[L]
        num_paths       = int(inputs['num_paths'])
        num_links       = int(inputs['num_links'])
        num_queues      = int(inputs['num_queues'])

        # Assertions: verify all shapes are correct
        assert traffic.dim() == 1,         f"traffic: {traffic.shape}"
        assert packets.dim() == 1,         f"packets: {packets.shape}"
        assert capacity.dim() == 1,         f"capacity: {capacity.shape}"
        assert queue_size.dim() == 1,       f"queue_size: {queue_size.shape}"
        assert weight.dim() == 1,          f"weight: {weight.shape}"
        assert model.dim() == 1,           f"model: {model.shape}"
        assert policy.dim() == 1,           f"policy: {policy.shape}"
        assert priority.dim() == 1,         f"priority: {priority.shape}"
        assert length.dim() == 1,           f"length: {length.shape}"
        assert num_paths > 0 and num_links > 0 and num_queues > 0, \
            f"num_paths={num_paths}, num_links={num_links}, num_queues={num_queues}"

        model_onehot    = F.one_hot(model.long(),     num_classes=self.max_num_models).float()  # (N, 7)
        policy_onehot   = F.one_hot(policy.long(),   num_classes=self.num_policies).float()   # (L, 4)
        priority_onehot = F.one_hot(priority.long(), num_classes=self.max_num_queues).float()  # (Q, 3)
        assert model_onehot.dim()    == 2, f"model_onehot: {model_onehot.shape}"
        assert policy_onehot.dim()   == 2, f"policy_onehot: {policy_onehot.shape}"
        assert priority_onehot.dim() == 2, f"priority_onehot: {priority_onehot.shape}"

        # ── Initial embeddings ───────────────────────────────────────────────────
        path_input = torch.cat([
            self._z_norm(traffic.unsqueeze(1),   'traffic'),
            self._z_norm(packets.unsqueeze(1),   'packets'),
            model_onehot,
            self._z_norm(eq_lambda.unsqueeze(1),  'eq_lambda'),
            self._z_norm(avg_pkts_lambda.unsqueeze(1), 'avg_pkts_lambda'),
            self._z_norm(exp_max_factor.unsqueeze(1),  'exp_max_factor'),
            self._z_norm(pkts_lambda_on.unsqueeze(1),  'pkts_lambda_on'),
            self._z_norm(avg_t_off.unsqueeze(1),   'avg_t_off'),
            self._z_norm(avg_t_on.unsqueeze(1),    'avg_t_on'),
            self._z_norm(ar_a.unsqueeze(1),        'ar_a'),
            self._z_norm(sigma.unsqueeze(1),        'sigma'),
        ], dim=1)
        path_state = self.path_embedding(path_input)   # (N, P=32)

        link_load = torch.zeros(num_links, device=device, dtype=path_state.dtype)
        for p_idx, link_ids in enumerate(link_to_path):
            for lid in link_ids:
                link_load[lid] += traffic[p_idx]
        link_load = link_load / (capacity + 1e-8)                    # (L,)
        link_state = self.link_embedding(
            torch.cat([link_load.unsqueeze(1), policy_onehot], dim=1))  # (L, 5) → (L, 32)
        assert link_state.dim() == 2 and link_state.shape[1] == self.link_state_dim, \
            f"link_state after embedding: {link_state.shape}"

        queue_input = torch.cat([
            self._z_norm(queue_size.unsqueeze(1), 'queue_size'),
            priority_onehot,
            weight.unsqueeze(1),
        ], dim=1)
        queue_state = self.queue_embedding(queue_input)   # (Q, 5) → (Q, 32)

        pkt_size = traffic / packets.clamp(min=1e-8)      # (N,)

        # ── Build padded tensors for path GRU ────────────────────────────────────
        max_hops = max(int(length.max().item()), 1)

        queue_to_path_pad = torch.full((num_paths, max_hops), -1,
                                       dtype=torch.long, device=device)
        link_to_path_pad  = torch.full((num_paths, max_hops), -1,
                                       dtype=torch.long, device=device)
        path_mask = torch.zeros((num_paths, max_hops), dtype=torch.float32, device=device)
        for p in range(num_paths):
            n_hops = min(len(queue_to_path[p]), len(link_to_path[p]))
            for h in range(n_hops):
                queue_to_path_pad[p, h] = queue_to_path[p][h]
                link_to_path_pad[p, h]  = link_to_path[p][h]
                path_mask[p, h] = 1.0

        cap_gather = capacity[link_to_path_pad].clamp_(min=1e-8)  # (N, H)

        # ── Message-passing iterations ──────────────────────────────────────────
        for it in range(self.iterations):
            # ══ Step 1: Link & Queue TO Path (sequential GRU, masked) ══════════
            # path_state_seq: (N, H+1, P) — position 0 = initial embedding
            path_state_seq = torch.zeros(
                num_paths, max_hops + 1, self.path_state_dim,
                dtype=torch.float32, device=device
            )
            path_state_seq[:, 0] = path_state

            h_prev = path_state
            for h in range(1, max_hops + 1):
                valid = path_mask[:, h - 1]
                if valid.sum() == 0:
                    continue

                q_in = queue_state[queue_to_path_pad[:, h - 1]]   # (N, Qdim)
                l_in = link_state[link_to_path_pad[:, h - 1]]    # (N, Ldim)
                combined = torch.cat([q_in, l_in], dim=1)         # (N, Q+L)

                valid_idx = valid.bool()
                updated = self.path_gru(combined[valid_idx], h_prev[valid_idx])  # (N_valid, P)
                h_prev = h_prev.clone()
                h_prev[valid_idx] = updated
                path_state_seq[valid_idx, h] = updated

            path_state = h_prev

            # ══ Step 2: Path TO Queue (ragged gather_nd + sum, matching TF) ══
            # path_to_queue[q_] = [[(pid, pos), ...], ...] per queue
            # TF: path_gather = gather_nd(path_state_seq, path_to_queue); path_sum = reduce_sum(...)
            p2q_flat_pid = []
            p2q_flat_pos = []
            p2q_flat_dst = []
            for qid in range(num_queues):
                for pid, pos in path_to_queue[qid]:
                    p2q_flat_pid.append(pid)
                    p2q_flat_pos.append(pos + 1)   # +1: position 0 is the initial embedding
                    p2q_flat_dst.append(qid)

            if p2q_flat_pid:
                p2q_pid_t = torch.tensor(p2q_flat_pid, dtype=torch.long, device=device)
                p2q_pos_t = torch.tensor(p2q_flat_pos, dtype=torch.long, device=device)
                p2q_dst_t = torch.tensor(p2q_flat_dst, dtype=torch.long, device=device)
                gathered = path_state_seq[p2q_pid_t, p2q_pos_t]               # (total, P)
                path_sum = torch.zeros(num_queues, self.path_state_dim,
                                      device=device, dtype=torch.float32)
                path_sum.index_add_(0, p2q_dst_t, gathered)                  # (Q, P)
            else:
                path_sum = torch.zeros(num_queues, self.path_state_dim,
                                      device=device, dtype=torch.float32)

            queue_state = self.queue_gru(path_sum, queue_state)              # (Q, Qdim)
            assert queue_state.dim() == 2 and queue_state.shape[1] == self.queue_state_dim, \
                f"queue_state after queue GRU iter {it}: {queue_state.shape}"

            # ══ Step 3: Queue TO Link — exact match to TF RNN(link_update, return_sequences=False) ══
            # TF logic: for each link, gather its queue states in order, feed sequentially through
            #           GRUCell(queue_state_dim) starting from current link_state, take final hidden state.
            #           For links with empty queue list, GRU still runs once on a zero vector.
            new_link_state = []
            for lid in range(num_links):
                qs = queue_to_link[lid]
                h = link_state[lid]                                      # (Ldim,)
                if qs:
                    # Gather queue states in arrival order, feed one-by-one into GRUCell
                    for qid in qs:
                        q_vec = queue_state[qid]                           # (Qdim,)
                        h = self.link_gru(q_vec.unsqueeze(0), h.unsqueeze(0)).squeeze(0)  # (Ldim,)
                else:
                    # Empty queue list: GRU still processes a zero vector (TF behavior)
                    zero_q = torch.zeros(self.queue_state_dim, device=device, dtype=torch.float32)
                    h = self.link_gru(zero_q.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
                new_link_state.append(h)
            link_state = torch.stack(new_link_state, dim=0)                # (L, Ldim)

            assert link_state.dim() == 2 and link_state.shape == (num_links, self.link_state_dim), \
                f"link_state after link GRU iter {it}: {link_state.shape}"

        # ── Readout ─────────────────────────────────────────────────────────────
        occupancy = self.readout_path(path_state_seq[:, 1:]).squeeze(-1)   # (N, H)
        occ_valid = occupancy * path_mask                                       # zero out padded
        queue_delay = (occ_valid / cap_gather).sum(dim=1)                      # (N,)

        pkt_exp = pkt_size.unsqueeze(1).expand(-1, max_hops)
        trans_delay = (pkt_exp / cap_gather).sum(dim=1)                        # (N,)

        return queue_delay + trans_delay

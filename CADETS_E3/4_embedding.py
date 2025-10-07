import argparse
import os
import logging
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from torch_geometric.data import TemporalData
from tqdm import tqdm

from config import *
from kairos_utils import *

# Setting for logging
logger = logging.getLogger("embedding_logger")
logger.setLevel(logging.INFO)
log_path = os.path.join(ARTIFACT_DIR, 'embedding.log')
os.makedirs(os.path.dirname(log_path), exist_ok=True)
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def path2higlist(p):
    l=[]
    spl=p.strip().split('/')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'/'+i)
        else:
            l.append(i)
    return l

def ip2higlist(p):
    l=[]
    spl=p.strip().split('.')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'.'+i)
        else:
            l.append(i)
    return l

def list2str(l):
    s=''
    for i in l:
        s+=i
    return s

def gen_feature(cur):
    # Firstly obtain all node labels
    nodeid2msg = gen_nodeid2msg(cur=cur)

    # Construct the hierarchical representation for each node label
    node_msg_dic_list = []
    for i in tqdm(nodeid2msg.keys()):
        if type(i) == int:
            higlist = []
            if 'netflow' in nodeid2msg[i].keys():
                higlist = ['netflow']
                higlist += ip2higlist(nodeid2msg[i]['netflow'])

            if 'file' in nodeid2msg[i].keys():
                higlist = ['file']
                higlist += path2higlist(nodeid2msg[i]['file'])

            if 'subject' in nodeid2msg[i].keys():
                higlist = ['subject']
                higlist += path2higlist(nodeid2msg[i]['subject'])
            if higlist:
                node_msg_dic_list.append(list2str(higlist))

    # Featurize the hierarchical node labels
    FH_string = FeatureHasher(n_features=node_embedding_dim, input_type="string")
    node2higvec=[]
    for i in tqdm(node_msg_dic_list):
        vec=FH_string.transform([[i]]).toarray()
        node2higvec.append(vec)
    node2higvec = np.array(node2higvec).reshape([-1, node_embedding_dim])
    torch.save(node2higvec, ARTIFACT_DIR + "node2higvec")
    return node2higvec

def gen_relation_onehot():
    relvec=torch.nn.functional.one_hot(torch.arange(0, len(rel2id.keys())//2), num_classes=len(rel2id.keys())//2)
    rel2vec={}
    for i in rel2id.keys():
        if type(i) is not int:
            rel2vec[i]= relvec[rel2id[i]-1]
            rel2vec[relvec[rel2id[i]-1]]=i
    torch.save(rel2vec, ARTIFACT_DIR + "rel2vec")
    return rel2vec

def _safe_label(label: str) -> str:
    sanitized = label.replace(":", "-").replace(" ", "_")
    allowed = []
    for ch in sanitized:
        if ch.isalnum() or ch in {'_', '-', '.'}:
            allowed.append(ch)
    return ''.join(allowed) or 'window'


def _build_dataset(edge_list: Sequence[Tuple[int, int, str, int]],
                   node2higvec: np.ndarray,
                   rel2vec,
                   logger,
                   label: str) -> Optional[TemporalData]:
    if not edge_list:
        logger.warning(f'No edges found for window "{label}". Skipping dataset creation.')
        return None

    dataset = TemporalData()
    src: List[int] = []
    dst: List[int] = []
    msg_tensors: List[torch.Tensor] = []
    timestamps: List[int] = []

    for src_id, dst_id, relation, timestamp in edge_list:
        src.append(src_id)
        dst.append(dst_id)
        timestamps.append(timestamp)
        msg_tensors.append(
            torch.cat([
                torch.from_numpy(node2higvec[src_id]),
                rel2vec[relation],
                torch.from_numpy(node2higvec[dst_id])
            ])
        )

    dataset.src = torch.tensor(src, dtype=torch.long)
    dataset.dst = torch.tensor(dst, dtype=torch.long)
    dataset.t = torch.tensor(timestamps, dtype=torch.long)
    dataset.msg = torch.vstack(msg_tensors).to(torch.float)
    torch.save(dataset, os.path.join(GRAPHS_DIR, f"graph_{label}.TemporalData.simple"))
    logger.info(f'Saved dataset graph_{label}.TemporalData.simple with {len(edge_list)} edges.')
    return dataset


def _fetch_edges(cur, start_ts: int, end_ts: int) -> List[Tuple[int, int, str, int]]:
    sql = f"""
        select * from event_table
        where timestamp_rec>'{start_ts}' and timestamp_rec<'{end_ts}'
        ORDER BY timestamp_rec;
    """
    cur.execute(sql)
    events = cur.fetchall()
    edge_list: List[Tuple[int, int, str, int]] = []
    for event in events:
        src = int(event[1])
        dst = int(event[4])
        relation = event[2]
        timestamp = int(event[5])
        if relation in include_edge_type:
            edge_list.append((src, dst, relation, timestamp))
    return edge_list


def gen_vectorized_graphs(cur,
                          node2higvec,
                          rel2vec,
                          logger,
                          days: Optional[Iterable[int]] = None) -> None:
    day_list = list(days) if days is not None else list(range(2, 14))
    for day in tqdm(day_list):
        start_timestamp = datetime_to_ns_time_US(f'2018-04-{day:02d} 00:00:00')
        end_timestamp = datetime_to_ns_time_US(f'2018-04-{day + 1:02d} 00:00:00')
        edges = _fetch_edges(cur, start_timestamp, end_timestamp)
        logger.info(f'2018-04-{day:02d}, edge list len: {len(edges)}')
        _build_dataset(edges, node2higvec, rel2vec, logger, label=f'4_{day}')


def gen_vectorized_window(cur,
                          node2higvec,
                          rel2vec,
                          logger,
                          start_ts: int,
                          end_ts: int,
                          label: str) -> None:
    if end_ts <= start_ts:
        raise ValueError("end_ts must be greater than start_ts")
    edges = _fetch_edges(cur, start_ts, end_ts)
    logger.info(f'Window {label}: edge list len: {len(edges)}')
    _build_dataset(edges, node2higvec, rel2vec, logger, label=_safe_label(label))


def _parse_time_arg(value: str) -> int:
    value = value.strip()
    if value.isdigit():
        return int(value)
    return datetime_to_ns_time_US(value)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate temporal graph embeddings.')
    parser.add_argument('--start', type=str,
                        help="Window start time in 'YYYY-MM-DD HH:MM:SS' (US/Eastern) or ns timestamp.")
    parser.add_argument('--end', type=str,
                        help="Window end time in 'YYYY-MM-DD HH:MM:SS' (US/Eastern) or ns timestamp.")
    parser.add_argument('--label', type=str,
                        help='Suffix used when saving a custom window dataset.')
    parser.add_argument('--days', type=int, nargs='+',
                        help='Specific April day numbers (e.g. 4 5 6) to materialize. Defaults to 2-13.')
    parser.add_argument('--skip-daily', action='store_true',
                        help='Do not generate per-day datasets when --start/--end is given.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logger.info("Start logging.")

    os.makedirs(GRAPHS_DIR, exist_ok=True)

    cur, _ = init_database_connection()
    node2higvec = gen_feature(cur=cur)
    rel2vec = gen_relation_onehot()

    custom_window_requested = args.start is not None or args.end is not None
    if custom_window_requested:
        if not (args.start and args.end):
            raise ValueError('Both --start and --end must be provided to generate a custom window.')
        start_ts = _parse_time_arg(args.start)
        end_ts = _parse_time_arg(args.end)
        label = args.label or f'{args.start}_to_{args.end}'
        gen_vectorized_window(cur=cur,
                              node2higvec=node2higvec,
                              rel2vec=rel2vec,
                              logger=logger,
                              start_ts=start_ts,
                              end_ts=end_ts,
                              label=label)

    generate_daily = (not custom_window_requested) or not args.skip_daily
    if generate_daily:
        gen_vectorized_graphs(cur=cur,
                              node2higvec=node2higvec,
                              rel2vec=rel2vec,
                              logger=logger,
                              days=args.days)

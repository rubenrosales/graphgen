import random
import time
from dataclasses import dataclass
from statistics import mean


@dataclass
class Community:
    id: str
    nodes: list[str]
    edges: list[tuple[str, str]]


class GraphStore:
    def __init__(self, data, op_latency_sec: float = 0.0):
        self.data = data
        self.op_latency_sec = op_latency_sec
        self.calls = {
            "get_node": 0,
            "get_edge": 0,
            "get_neighbors": 0,
            "get_nodes_by_ids": 0,
            "get_edges_by_pairs": 0,
        }

    def reset_calls(self):
        for k in self.calls:
            self.calls[k] = 0

    def _tick(self, key: str):
        self.calls[key] += 1
        if self.op_latency_sec > 0:
            time.sleep(self.op_latency_sec)


def build_graph(node_count: int = 5000, edge_count: int = 20000):
    g = {"nodes": {}, "edges": {}, "adj": {}}
    for i in range(node_count):
        nid = f"n{i}"
        g["nodes"][nid] = {"entity_name": nid, "description": "node"}
        g["adj"][nid] = []
    for _ in range(edge_count):
        u = f"n{random.randint(0, node_count - 1)}"
        v = f"n{random.randint(0, node_count - 1)}"
        if u == v:
            continue
        key = tuple(sorted((u, v)))
        g["edges"][key] = {"description": "edge"}
        g["adj"][u].append(v)
        g["adj"][v].append(u)
    return g


def get_node(g: GraphStore, n):
    g._tick("get_node")
    return g.data["nodes"].get(n)


def get_edge(g: GraphStore, u, v):
    g._tick("get_edge")
    return g.data["edges"].get(tuple(sorted((u, v))))


def get_neighbors(g: GraphStore, n):
    g._tick("get_neighbors")
    return g.data["adj"].get(n, [])


def get_nodes_by_ids(g: GraphStore, node_ids):
    g._tick("get_nodes_by_ids")
    return {n: g.data["nodes"][n] for n in node_ids if n in g.data["nodes"]}


def get_edges_by_pairs(g: GraphStore, edge_pairs):
    g._tick("get_edges_by_pairs")
    out = {}
    for u, v in edge_pairs:
        ed = get_edge(g, u, v)
        if ed is not None:
            out[(u, v)] = ed
    return out


def old_community2batch(comm: Community, g):
    nodes_data = []
    for node in comm.nodes:
        node_data = get_node(g, node)
        if node_data:
            nodes_data.append((node, node_data))

    edges_data = []
    for edge in comm.edges:
        if not isinstance(edge, tuple) or len(edge) != 2:
            continue
        u, v = edge
        if u == v:
            continue
        edge_data = get_edge(g, u, v)
        if edge_data:
            edges_data.append((u, v, edge_data))
    return nodes_data, edges_data


def new_community2batch(comm: Community, g):
    nodes_lookup = get_nodes_by_ids(g, comm.nodes) if comm.nodes else {}
    nodes_data = [(node, nodes_lookup[node]) for node in comm.nodes if node in nodes_lookup]
    valid_edges = []
    for edge in comm.edges:
        if isinstance(edge, tuple) and len(edge) == 2 and edge[0] != edge[1]:
            valid_edges.append(edge)
    edge_lookup = get_edges_by_pairs(g, valid_edges) if valid_edges else {}
    edges_data = [(u, v, edge_lookup[(u, v)]) for u, v in valid_edges if (u, v) in edge_lookup]
    return nodes_data, edges_data


def benchmark_community2batch(g, rounds: int = 20, op_latency_sec: float = 0.0):
    store = GraphStore(g, op_latency_sec=op_latency_sec)
    nodes = list(g["nodes"].keys())
    communities = []
    for i in range(rounds):
        picked = random.sample(nodes, 30)
        edges = []
        for n in picked[:10]:
            for nb in get_neighbors(store, n)[:3]:
                edges.append(tuple(sorted((n, nb))))
        communities.append(Community(id=f"c{i}", nodes=picked, edges=edges))

    old_times = []
    new_times = []
    old_calls = []
    new_calls = []
    for comm in communities:
        store.reset_calls()
        t0 = time.perf_counter()
        old_community2batch(comm, store)
        old_times.append(time.perf_counter() - t0)
        old_calls.append(dict(store.calls))

        store.reset_calls()
        t1 = time.perf_counter()
        new_community2batch(comm, store)
        new_times.append(time.perf_counter() - t1)
        new_calls.append(dict(store.calls))

    old_avg_calls = {
        k: int(mean(c[k] for c in old_calls))
        for k in old_calls[0]
    }
    new_avg_calls = {
        k: int(mean(c[k] for c in new_calls))
        for k in new_calls[0]
    }
    return mean(old_times), mean(new_times), old_avg_calls, new_avg_calls


def benchmark_traversal_pattern(g, rounds: int = 5):
    store = GraphStore(g)
    nodes = list(g["nodes"].keys())

    old_times = []
    new_times = []
    edges = [(u, v, d) for (u, v), d in g["edges"].items()]
    adjacency = {}
    for u, v, _ in edges:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
    for _ in range(rounds):
        seeds = random.sample(nodes, 1000)

        t0 = time.perf_counter()
        _ = [(s, get_neighbors(store, s)) for s in seeds for _ in range(5)]
        old_times.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        _ = [(s, adjacency.get(s, [])) for s in seeds for _ in range(5)]
        new_times.append(time.perf_counter() - t1)

    return mean(old_times), mean(new_times)


def benchmark_bfs_like(g, rounds: int = 5):
    times = []
    nodes = list(g["nodes"].keys())
    edges = [(u, v, d) for (u, v), d in g["edges"].items()]
    for _ in range(rounds):
        t0 = time.perf_counter()
        adjacency = {}
        for u, v, _ in edges:
            adjacency.setdefault(u, []).append(v)
            adjacency.setdefault(v, []).append(u)
        visited = set()
        for seed in random.sample(nodes, min(1000, len(nodes))):
            if seed in visited:
                continue
            stack = [seed]
            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                for nb in adjacency.get(n, [])[:5]:
                    if nb not in visited:
                        stack.append(nb)
        times.append(time.perf_counter() - t0)
    return mean(times)


def main():
    random.seed(42)
    g = build_graph()

    old_c2b, new_c2b, old_calls, new_calls = benchmark_community2batch(g)
    old_c2b_io, new_c2b_io, _, _ = benchmark_community2batch(
        g, op_latency_sec=0.0002
    )
    old_tr, new_tr = benchmark_traversal_pattern(g)
    bfs_t = benchmark_bfs_like(g)

    def speedup(old, new):
        return old / max(new, 1e-12)

    print("=== GraphGen Quick Benchmark ===")
    print(f"community2batch_old_avg_s: {old_c2b:.6f}")
    print(f"community2batch_new_avg_s: {new_c2b:.6f}")
    print(f"community2batch_speedup_x: {speedup(old_c2b, new_c2b):.2f}")
    print(f"community2batch_old_calls: {old_calls}")
    print(f"community2batch_new_calls: {new_calls}")
    print(f"community2batch_simulated_io_old_avg_s: {old_c2b_io:.6f}")
    print(f"community2batch_simulated_io_new_avg_s: {new_c2b_io:.6f}")
    print(
        f"community2batch_simulated_io_speedup_x: {speedup(old_c2b_io, new_c2b_io):.2f}"
    )
    print(f"traversal_old_avg_s: {old_tr:.6f}")
    print(f"traversal_new_avg_s: {new_tr:.6f}")
    print(f"traversal_speedup_x: {speedup(old_tr, new_tr):.2f}")
    print(f"bfs_like_current_avg_s: {bfs_t:.6f}")


if __name__ == "__main__":
    main()

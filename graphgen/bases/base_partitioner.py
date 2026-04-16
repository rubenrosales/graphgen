from abc import ABC, abstractmethod
from typing import Any, List

from graphgen.bases.base_storage import BaseGraphStorage
from graphgen.bases.datatypes import Community


class BasePartitioner(ABC):
    @abstractmethod
    def partition(
        self,
        g: BaseGraphStorage,
        **kwargs: Any,
    ) -> List[Community]:
        """
        Graph -> Communities
        :param g: Graph storage instance
        :param kwargs: Additional parameters for partitioning
        :return: List of communities
        """

    @staticmethod
    def community2batch(
        comm: Community, g: BaseGraphStorage
    ) -> tuple[
        list[tuple[str, dict]], list[tuple[Any, Any, dict] | tuple[Any, Any, Any]]
    ]:
        """
        Convert communities to batches of nodes and edges.
        :param comm: Community
        :param g: Graph storage instance
        :return: List of batches, each batch is a tuple of (nodes, edges)
        """
        nodes = comm.nodes
        edges = comm.edges
        nodes_lookup = g.get_nodes_by_ids(nodes) if nodes else {}
        nodes_data = [(node, nodes_lookup[node]) for node in nodes if node in nodes_lookup]
        valid_edges = []
        for edge in edges:
            if not isinstance(edge, tuple) or len(edge) != 2:
                continue
            u, v = edge
            if u == v:
                continue
            valid_edges.append((u, v))
        edge_lookup = g.get_edges_by_pairs(valid_edges) if valid_edges else {}
        edges_data = [(u, v, edge_lookup[(u, v)]) for u, v in valid_edges if (u, v) in edge_lookup]
        return nodes_data, edges_data

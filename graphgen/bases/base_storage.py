from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, List, Set, TypeVar, Union

T = TypeVar("T")


@dataclass
class StorageNameSpace:
    working_dir: str = None
    namespace: str = None

    def index_done_callback(self):
        """commit the storage operations after indexing"""

    def query_done_callback(self):
        """commit the storage operations after querying"""


class BaseKVStorage(Generic[T], StorageNameSpace):
    def all_keys(self) -> list[str]:
        raise NotImplementedError

    def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        raise NotImplementedError

    def get_all(self) -> dict[str, T]:
        raise NotImplementedError

    def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    def update(self, data: dict[str, T]):
        raise NotImplementedError

    def delete(self, ids: list[str]):
        raise NotImplementedError

    def drop(self):
        raise NotImplementedError

    def reload(self):
        raise NotImplementedError


class BaseGraphStorage(StorageNameSpace, ABC):
    @abstractmethod
    def is_directed(self) -> bool:
        pass

    @abstractmethod
    def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_all_node_degrees(self) -> Dict[str, int]:
        pass

    def get_isolated_nodes(self) -> List[str]:
        return [
            node_id
            for node_id, degree in self.get_all_node_degrees().items()
            if degree == 0
        ]

    @abstractmethod
    def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    @abstractmethod
    def update_node(self, node_id: str, node_data: dict[str, any]):
        raise NotImplementedError

    @abstractmethod
    def get_all_nodes(self) -> Union[list[tuple[str, dict]], None]:
        raise NotImplementedError

    @abstractmethod
    def get_node_count(self) -> int:
        pass

    @abstractmethod
    def get_edge(self, source_node_id: str, target_node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    @abstractmethod
    def update_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, any]
    ):
        raise NotImplementedError

    @abstractmethod
    def get_all_edges(self) -> Union[list[tuple[str, str, dict]], None]:
        raise NotImplementedError

    @abstractmethod
    def get_edge_count(self) -> int:
        pass

    @abstractmethod
    def get_node_edges(self, source_node_id: str) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError

    @abstractmethod
    def upsert_node(self, node_id: str, node_data: dict[str, any]):
        raise NotImplementedError

    @abstractmethod
    def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, any]
    ):
        raise NotImplementedError

    @abstractmethod
    def delete_node(self, node_id: str):
        raise NotImplementedError

    @abstractmethod
    def get_neighbors(self, node_id: str) -> List[str]:
        raise NotImplementedError

    def get_neighbors_batch(self, node_ids: List[str]) -> Dict[str, List[str]]:
        """Optional batch API with a safe default implementation."""
        return {node_id: self.get_neighbors(node_id) for node_id in node_ids}

    def get_nodes_by_ids(self, node_ids: List[str]) -> Dict[str, dict]:
        """Optional bulk node fetch API with a safe default implementation."""
        data: Dict[str, dict] = {}
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node is not None:
                data[node_id] = node
        return data

    def get_edges_by_pairs(
        self, edge_pairs: List[tuple[str, str]]
    ) -> Dict[tuple[str, str], dict]:
        """Optional bulk edge fetch API with a safe default implementation."""
        data: Dict[tuple[str, str], dict] = {}
        for u, v in edge_pairs:
            edge_data = self.get_edge(u, v) or self.get_edge(v, u)
            if edge_data is not None:
                data[(u, v)] = edge_data
        return data

    def upsert_nodes_bulk(self, nodes: Dict[str, dict]):
        """Optional bulk node upsert API with a safe default implementation."""
        for node_id, node_data in nodes.items():
            self.upsert_node(node_id, node_data)

    def upsert_edges_bulk(self, edges: List[tuple[str, str, dict]]):
        """Optional bulk edge upsert API with a safe default implementation."""
        for src_id, tgt_id, edge_data in edges:
            self.upsert_edge(src_id, tgt_id, edge_data)

    @abstractmethod
    def reload(self):
        raise NotImplementedError

    @abstractmethod
    def get_connected_components(self, undirected: bool = True) -> List[Set[str]]:
        raise NotImplementedError

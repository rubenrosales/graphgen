import os
from typing import Iterable, Tuple

from graphgen.bases import BaseGraphStorage, BaseOperator, BaseTokenizer
from graphgen.common.init_storage import init_storage
from graphgen.utils import logger


class PartitionService(BaseOperator):
    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        graph_backend: str = "kuzu",
        **partition_kwargs,
    ):
        super().__init__(
            working_dir=working_dir, kv_backend=kv_backend, op_name="partition"
        )
        self.kg_instance: BaseGraphStorage = init_storage(
            backend=graph_backend,
            working_dir=working_dir,
            namespace="graph",
        )
        tokenizer_model = os.getenv("TOKENIZER_MODEL", "cl100k_base")

        from graphgen.models import Tokenizer

        self.tokenizer_instance: BaseTokenizer = Tokenizer(model_name=tokenizer_model)
        method = partition_kwargs["method"]
        self.method_params = partition_kwargs.get("method_params", {})

        if method == "bfs":
            from graphgen.models import BFSPartitioner

            self.partitioner = BFSPartitioner()
        elif method == "dfs":
            from graphgen.models import DFSPartitioner

            self.partitioner = DFSPartitioner()
        elif method == "ece":
            # before ECE partitioning, we need to:
            # 'quiz' and 'judge' to get the comprehension loss if unit_sampling is not random
            from graphgen.models import ECEPartitioner

            self.partitioner = ECEPartitioner()
        elif method == "leiden":
            from graphgen.models import LeidenPartitioner

            self.partitioner = LeidenPartitioner()
        elif method == "anchor_bfs":
            from graphgen.models import AnchorBFSPartitioner

            self.partitioner = AnchorBFSPartitioner(
                anchor_type=self.method_params.get("anchor_type"),
                anchor_ids=set(self.method_params.get("anchor_ids", []))
                if self.method_params.get("anchor_ids")
                else None,
            )
        elif method == "triple":
            from graphgen.models import TriplePartitioner

            self.partitioner = TriplePartitioner()
        elif method == "quintuple":
            from graphgen.models import QuintuplePartitioner

            self.partitioner = QuintuplePartitioner()
        else:
            raise ValueError(f"Unsupported partition method: {method}")
        self._cached_communities = None
        self._cache_signature = None

    @staticmethod
    def _freeze(value):
        if isinstance(value, dict):
            return tuple((k, PartitionService._freeze(v)) for k, v in sorted(value.items()))
        if isinstance(value, list):
            return tuple(PartitionService._freeze(v) for v in value)
        if isinstance(value, set):
            return tuple(sorted(PartitionService._freeze(v) for v in value))
        return value

    def process(self, batch: list) -> Tuple[Iterable[dict], dict]:
        # this operator does not consume any batch data
        # but for compatibility we keep the interface
        self.kg_instance.reload()
        signature = (
            self.kg_instance.get_node_count(),
            self.kg_instance.get_edge_count(),
            self._freeze(self.method_params),
            self.partitioner.__class__.__name__,
        )
        if self._cache_signature != signature or self._cached_communities is None:
            self._cached_communities = list(
                self.partitioner.partition(g=self.kg_instance, **self.method_params)
            )
            self._cache_signature = signature

        communities: Iterable = iter(self._cached_communities)

        def generator():
            count = 0
            for community in communities:
                count += 1
                b = self.partitioner.community2batch(community, g=self.kg_instance)

                result = {
                    "nodes": b[0],
                    "edges": b[1],
                }
                result["_trace_id"] = self.get_trace_id(result)
                yield result
            logger.info("Total communities partitioned: %d", count)

        return generator(), {}

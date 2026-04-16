from collections import defaultdict
from typing import List

from graphgen.bases import BaseLLMWrapper
from graphgen.bases.base_storage import BaseGraphStorage
from graphgen.bases.datatypes import Chunk
from graphgen.models import LightRAGKGBuilder
from graphgen.utils import run_concurrent


def build_text_kg(
    llm_client: BaseLLMWrapper,
    kg_instance: BaseGraphStorage,
    chunks: List[Chunk],
    max_loop: int = 3,
) -> tuple:
    """
    :param llm_client: Synthesizer LLM model to extract entities and relationships
    :param kg_instance
    :param chunks
    :param max_loop: Maximum number of loops for entity and relationship extraction
    :return:
    """

    kg_builder = LightRAGKGBuilder(llm_client=llm_client, max_loop=max_loop)

    results = run_concurrent(
        kg_builder.extract,
        chunks,
        desc="[2/4]Extracting entities and relationships from chunks",
        unit="chunk",
    )
    results = [res for res in results if res]

    nodes = defaultdict(list)
    edges = defaultdict(list)
    for n, e in results:
        for k, v in n.items():
            nodes[k].extend(v)
        for k, v in e.items():
            edges[tuple(sorted(k))].extend(v)

    node_items = list(nodes.items())
    existing_nodes = kg_instance.get_nodes_by_ids([entity_name for entity_name, _ in node_items])
    nodes = run_concurrent(
        lambda kv: kg_builder.merge_nodes(
            kv,
            kg_instance=kg_instance,
            existing_node=existing_nodes.get(kv[0]),
            persist=False,
        ),
        node_items,
        desc="Inserting entities into storage",
    )
    kg_instance.upsert_nodes_bulk(
        {
            node["entity_name"]: node
            for node in nodes
            if node and node.get("entity_name")
        }
    )

    edge_items = list(edges.items())
    merged_node_ids = {node["entity_name"] for node in nodes if node and node.get("entity_name")}
    edges = run_concurrent(
        lambda kv: kg_builder.merge_edges(
            kv,
            kg_instance=kg_instance,
            existing_nodes=merged_node_ids,
            persist=False,
        ),
        edge_items,
        desc="Inserting relationships into storage",
    )
    kg_instance.upsert_edges_bulk(
        [
            (edge["src_id"], edge["tgt_id"], edge)
            for edge in edges
            if edge and edge.get("src_id") and edge.get("tgt_id")
        ]
    )

    return nodes, edges

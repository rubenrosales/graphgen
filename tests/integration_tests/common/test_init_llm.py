import sys
from unittest.mock import MagicMock

from graphgen.common.init_llm import LLMFactory


def test_create_llm_uses_local_wrapper_in_local_mode(monkeypatch):
    monkeypatch.setenv("GRAPHGEN_EXECUTION_MODE", "local")

    fake_wrapper = object()
    create_local = MagicMock(return_value=fake_wrapper)
    monkeypatch.setattr(LLMFactory, "_create_local_llm", create_local)

    result = LLMFactory.create_llm("synthesizer", "http_api", {"model": "fake-model"})

    assert result is fake_wrapper
    create_local.assert_called_once_with("http_api", {"model": "fake-model"})


def test_create_llm_skips_local_wrapper_outside_local_mode(monkeypatch):
    monkeypatch.delenv("GRAPHGEN_EXECUTION_MODE", raising=False)

    create_local = MagicMock()
    monkeypatch.setattr(LLMFactory, "_create_local_llm", create_local)

    fake_actor_handle = MagicMock()
    fake_actor_handle.ready.remote.return_value = "ready-ref"
    fake_proxy = object()

    fake_remote_builder = MagicMock()
    fake_remote_builder.options.return_value.remote.return_value = fake_actor_handle

    fake_ray = MagicMock()
    fake_ray.get_actor.side_effect = ValueError("missing actor")
    fake_ray.remote.return_value = fake_remote_builder
    fake_ray.get.return_value = True

    proxy_ctor = MagicMock(return_value=fake_proxy)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(
        "graphgen.common.init_llm.LLMServiceProxy",
        proxy_ctor,
    )

    result = LLMFactory.create_llm("synthesizer", "http_api", {"model": "fake-model"})

    assert result is fake_proxy
    create_local.assert_not_called()
    proxy_ctor.assert_called_once_with(fake_actor_handle)

"""CPU checks of the mounted profiler's argument and checkpoint control flow."""
import argparse
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

FORK = Path(os.environ.get("MEGATRON_FORK", "/vast/cs2731_2026f/wat29/Megatron-LM"))


@pytest.fixture
def main_ast():
    path = FORK / "profile_strategies.py"
    if not path.exists():
        pytest.skip("external Megatron fork unavailable")
    tree = ast.parse(path.read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def execute(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "profile_strategies.py", "exec"), namespace)


@pytest.mark.parametrize("disabled,count", [(False, 1), (False, 2), (True, 1), (True, 2)])
def test_checkpoint_free_menu_validation_before_megatron_init(main_ast, monkeypatch, disabled, count):
    import sys
    argv = ["profile_strategies.py", "--profile-menu", "menu.json", "--output", "out.json"]
    if disabled:
        argv.append("--profile-no-checkpoints")
    monkeypatch.setattr(sys, "argv", argv)
    namespace = dict(argparse=argparse, sys=sys, os=os, load_profile_menu=lambda path: [None] * count)
    prefix = []
    for node in main_ast.body:
        if isinstance(node, ast.ImportFrom):
            break
        prefix.append(node)
    if disabled and count != 1:
        with pytest.raises(SystemExit):
            execute(prefix, namespace)
        return
    execute(prefix, namespace)
    assert "--use-distributed-optimizer" in sys.argv
    assert ("--save" in sys.argv) is not disabled
    assert (namespace["work_ckpt"] is None) is disabled


@pytest.mark.parametrize("disabled,expected", [(False, 2), (True, 0)])
def test_both_save_sites_respect_opt_in(main_ast, disabled, expected):
    # Execute the actual save guards without importing CUDA/Megatron dependencies.
    save_guards = [node for node in ast.walk(main_ast) if isinstance(node, ast.If)
                   and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                           and child.func.id == "save_checkpoint" for child in ast.walk(node))]
    assert len(save_guards) == 2
    save = Mock()
    namespace = dict(profile_args=SimpleNamespace(profile_no_checkpoints=disabled),
                     is_first=True, rank=1, save_checkpoint=save,
                     model=object(), optimizer=object(), opt_param_scheduler=object(),
                     time=SimpleNamespace(perf_counter=lambda: 1.0))
    execute(save_guards, namespace)
    assert save.call_count == expected

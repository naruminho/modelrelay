# modelrelay — regras do repositório

- **Mudou o modelrelay, suba a versão** (pedido do Naruminho): `version` em `pyproject.toml` e `__version__` em
  `src/modelrelay/__init__.py` (os dois iguais; recurso novo sobe o do meio, correção sobe o último;
  `tests/test_version.py` confere). Depois do merge, uma release `vX.Y.Z` no GitHub publica no PyPI
  (`.github/workflows/publish.yml`); sem poder criar release (agente sem acesso a tags), rode o workflow `publish`
  à mão na main com `publicar: true`.
- **E atualize o sagadeck para exigir a versão nova** (repositório `naruminho/sagadeck`): `MIN_MODELRELAY` em
  `python/sagadeck/llm.py` (o sagadeck avisa quem estiver com um modelrelay mais velho e mostra o comando para
  atualizar) e o extra `ia` no `pyproject.toml` de lá (`modelrelay>=X.Y.Z`).
- Testes: `pytest` (a tela de configuração tem os testes em `tests/test_console.py`). Nenhuma mudança sem teste.

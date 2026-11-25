"""Configuracoes equivalentes por um eixo inerte (alta cardinalidade).

Nos modos de Transformer, o grid varia a estrategia de alta cardinalidade
(``minimum-frequency`` 10 ou ``hashing`` 1024). Ela so atua nas colunas
categoricas de alta cardinalidade. Numa base sem nenhuma delas (a ULB nao tem
categoricas; a ``home-credit-somente-aplicacao`` nao tem nenhuma acima dos
limites), as duas estrategias geram os mesmos splits e o mesmo vocabulario.

Duas configuracoes sao equivalentes quando o manifesto mostra:

- os mesmos ``content_md5`` (splits) e os mesmos ``vocabulary_md5``;
- a mesma identidade em todos os outros eixos;
- diferenca somente nos campos de alta cardinalidade.

O modelo recebe entradas byte a byte iguais, entao basta executar uma delas.
O representante e a configuracao com a estrategia padrao
(``minimum-frequency``), para que os resultados de todas as bases fiquem no
mesmo valor do eixo; as demais sao gemeas dele.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

HIGH_CARDINALITY_FIELDS = ("high_cardinality_strategy", "high_cardinality_parameter")
DEFAULT_HIGH_CARDINALITY_STRATEGY = "minimum-frequency"
TRANSFORMER_MODES = ("transformer-indexed", "sequence-tokenized")
# Eixos do grid que precisam coincidir para a equivalencia.
IDENTITY_FIELDS = (
    "dataset_id",
    "categorical_handling_mode",
    "categorical_encoding_strategy",
    "fill_na_strategy",
    "undersampling_technique",
    "undersampling_parameter_values",
    "test_strategy",
    "random_seed",
    "numeric_normalization",
)


@dataclass(frozen=True)
class EquivalentGroup:
    dataset_id: str
    representative: str
    twins: tuple[str, ...]
    axis_values: tuple[str, ...]


def _values(rows: pd.DataFrame, column: str) -> frozenset[str]:
    if column not in rows:
        return frozenset()
    return frozenset(str(value) for value in rows[column].dropna() if str(value).strip())


def inert_high_cardinality_groups(manifest: pd.DataFrame) -> list[EquivalentGroup]:
    """Grupos de configuracoes que diferem so pelo eixo de alta cardinalidade
    e produzem os mesmos splits e vocabularios."""
    if manifest.empty or "content_md5" not in manifest:
        return []
    candidates: dict[tuple, list[tuple[str, str]]] = {}
    for md5, rows in manifest.groupby("configuration_md5"):
        first = rows.iloc[0]
        if str(first.get("categorical_handling_mode", "")) not in TRANSFORMER_MODES:
            continue
        contents = _values(rows, "content_md5")
        vocabularies = _values(rows, "vocabulary_md5")
        if not contents or not vocabularies:
            continue
        identity = tuple(str(first.get(field, "")) for field in IDENTITY_FIELDS)
        axis = "/".join(str(first.get(field, "")) for field in HIGH_CARDINALITY_FIELDS)
        candidates.setdefault((identity, contents, vocabularies), []).append((str(md5), axis))
    groups = []
    for (identity, _contents, _vocabularies), members in candidates.items():
        if len({axis for _md5, axis in members}) < 2:
            continue
        members = sorted(
            members,
            key=lambda member: (
                not member[1].startswith(f"{DEFAULT_HIGH_CARDINALITY_STRATEGY}/"),
                member[0],
            ),
        )
        groups.append(
            EquivalentGroup(
                dataset_id=identity[0],
                representative=members[0][0],
                twins=tuple(md5 for md5, _axis in members[1:]),
                axis_values=tuple(axis for _md5, axis in members),
            )
        )
    return sorted(groups, key=lambda group: (group.dataset_id, group.representative))


def twin_map(groups: list[EquivalentGroup]) -> dict[str, str]:
    """``configuration_md5`` gemea -> representante."""
    return {twin: group.representative for group in groups for twin in group.twins}


def groups_table(groups: list[EquivalentGroup]) -> pd.DataFrame:
    """Tabela do apendice: uma linha por gemea."""
    return pd.DataFrame(
        [
            {
                "dataset_id": group.dataset_id,
                "representative_configuration_md5": group.representative,
                "twin_configuration_md5": twin,
                "eixo_inerte": "high_cardinality_strategy/high_cardinality_parameter",
                "valores_do_eixo": "; ".join(group.axis_values),
            }
            for group in groups
            for twin in group.twins
        ],
        columns=[
            "dataset_id",
            "representative_configuration_md5",
            "twin_configuration_md5",
            "eixo_inerte",
            "valores_do_eixo",
        ],
    )

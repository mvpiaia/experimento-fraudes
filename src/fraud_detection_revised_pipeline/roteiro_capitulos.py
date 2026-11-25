"""Roteiro da reescrita dos Capítulos 4 a 6 da dissertação.

Percorre os ``\\todo{...}`` de ``resultados.tex``, ``discussao_resultados.tex`` e
``conclusoes.tex`` e associa cada pendência à saída gerada que a responde
(figura, tabela ``.tex`` ou chave de um JSON de resumo). Indica, para cada
saída, se ela já existe nos diretórios informados. A prosa continua escrita
na dissertação; o roteiro garante que cada afirmação tenha uma saída gerada
que a sustente e que as Conclusões e os Trabalhos futuros retomem os
desfechos e as divergências registradas.

Grava ``roteiro_capitulos.md``. Só lê arquivos.

Uso::

    python3 roteiro_capitulos.py --thesis-directory <eqm_paper> \\
        --output-directories <eqm_paper>/figuras <eqm_paper>/tabelas analise_hipoteses \\
        --output roteiro_capitulos.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

CHAPTER_FILES = ("resultados.tex", "discussao_resultados.tex", "conclusoes.tex")
SIGLAS = {"aucroc": "AUC-ROC", "aucpr": "AUC-PR", "devnet": "DevNet", "woe": "WoE", "bigan": "BiGAN"}

# Regras avaliadas em ordem: (padrão no texto da pendência ou rótulo da seção, saídas).
RULES: list[tuple[str, tuple[str, ...]]] = [
    (r"As seções deste capítulo|Este capítulo (reúne|será)|Esta seção confronta", ("(abertura: só texto)",)),
    (r"sec:resultados-execucoes|execuções previstas", ("tab_execucoes.tex",)),
    (r"estrato|histórico|C3\.19|Q2", ("history_strata_analysis.py (tabela LaTeX)",)),
    (r"sec:resultados-classicos|regressão logística, do Naive Bayes",
     ("discussao_desempenho_classicos.png", "discussao_balanceamento_classicos.png",
      "discussao_codificacao_classicos.png", "discussao_imputacao.png")),
    (r"sec:resultados-boosting|CatBoost e do XGBoost por base",
     ("discussao_desempenho_boosting.png", "discussao_imputacao.png")),
    (r"sec:resultados-anomalias|Isolation Forest e do DevNet",
     ("discussao_desempenho_anomalia.png", "discussao_anomalia_rotulos.png")),
    (r"sec:resultados-transformers|FT-Transformer e do TabBERT",
     ("discussao_desempenho_transformers.png", "discussao_transformers.png")),
    (r"sec:resultados-ablacao|identidade|H4", ("tab_h4.tex", "resumo_h4.json:H4")),
    (r"sec:resultados-custo|Tempos de preparação", ("build_reproducibility_tables.py (Apêndice B)", "q3_custo.csv")),
    (r"sem dados sintéticos|RUS\+SMOTE|H1", ("tab_h1.tex", "resumo_hipoteses.json:H1")),
    (r"H3|restrita às configurações sem dados sintéticos", ("tab_h3.tex", "resumo_hipoteses.json:H3")),
    (r"Posição de cada algoritmo|ranking|Friedman|Robustez entre fontes|H2", ("tab_h2.tex", "resumo_hipoteses.json:H2")),
    (r"holdout\}? temporal|Q1", ("tab_q1.tex", "discussao_protocolo.png", "resumo_hipoteses.json:Q1")),
    (r"nativo de categorias|Q3", ("tab_q3.tex", "resumo_hipoteses.json:Q3")),
    (r"Comparação entre bases|comparação entre bases", ("discussao_desempenho_classicos.png", "discussao_desempenho_boosting.png")),
    (r"cap:conclusoes|Síntese|Contribuições|Limitações|Trabalhos futuros|desfechos definidos",
     ("resumo_hipoteses.json", "resumo_h4.json", "figuras_discussao_resumo.json")),
    (r"G15", ("discussao_codificacao_classicos.png", "discussao_imputacao.png")),
]


def pending_items(thesis: Path) -> list[dict]:
    items = []
    for name in CHAPTER_FILES:
        path = thesis / name
        if not path.exists():
            continue
        section = name
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            label = re.search(r"\\label\{([^}]*)\}", line)
            if re.match(r"\s*\\(chapter|section|subsection)", line) and label:
                section = label.group(1)
            for match in re.finditer(r"\\todo\{([^}]*)\}", line):
                text = re.sub(r"\\ac[slpf]*\{([^}]*)\}", lambda m: SIGLAS.get(m.group(1), m.group(1).upper()),
                              line[match.end():])
                text = re.sub(r"\\(textit|emph|textbf)\{([^}]*)\}", r"\2", text)
                text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", text)
                text = re.sub(r"[{}~$]", " ", text)
                text = re.sub(r"\s+", " ", text).strip()[:160]
                items.append({"arquivo": name, "linha": number, "secao": section, "marca": match.group(1), "texto": text})
    return items


def outputs_for(item: dict) -> tuple[str, ...]:
    haystack = f"{item['secao']} {item['marca']} {item['texto']}"
    for pattern, outputs in RULES:
        if re.search(pattern, haystack):
            return outputs
    return ()


def exists(output: str, directories: list[Path]) -> bool:
    if output.startswith("("):
        return True
    name = output.split(":")[0].split(" (")[0]
    if name.endswith(".py"):
        return True  # script existente; a saída depende de execução própria
    return any((directory / name).exists() for directory in directories)


def render(items: list[dict], directories: list[Path]) -> str:
    lines = ["# Roteiro da reescrita dos Capítulos 4–6", "",
             "Cada pendência `\\todo` da dissertação, com a saída gerada que a responde.", "",
             "| Arquivo:linha | Seção | Pendência | Saídas | Situação |", "|---|---|---|---|---|"]
    without = 0
    for item in items:
        outputs = outputs_for(item)
        if not outputs:
            without += 1
            status = "sem saída associada"
        else:
            missing = [o for o in outputs if not exists(o, directories)]
            status = "gerada" if not missing else "pendente: " + ", ".join(missing)
        text = item["texto"].replace("|", "/")
        lines.append(f"| {item['arquivo']}:{item['linha']} | `{item['secao']}` | [{item['marca']}] {text} | "
                     f"{', '.join(outputs) or '—'} | {status} |")
    lines += ["", f"Pendências: {len(items)}; sem saída associada: {without}."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--thesis-directory", required=True)
    parser.add_argument("--output-directories", nargs="*", default=[])
    parser.add_argument("--output", default="roteiro_capitulos.md")
    arguments = parser.parse_args()
    directories = [Path(d) for d in arguments.output_directories]
    items = pending_items(Path(arguments.thesis_directory))
    Path(arguments.output).write_text(render(items, directories), encoding="utf-8")
    print(f"{len(items)} pendências; roteiro em {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

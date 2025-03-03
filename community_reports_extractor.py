# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""
Reference:
 - [graphrag](https://github.com/microsoft/graphrag)
"""

import logging
import json
import re
import traceback
from typing import Callable, Optional
from dataclasses import dataclass
import networkx as nx
import pandas as pd
import tiktoken

import leiden
from community_report_prompt import COMMUNITY_REPORT_PROMPT
from leiden import add_community_info2graph
from utils import ErrorHandlerFn, perform_variable_replacements, dict_has_keys_with_types
from timeit import default_timer as timer
from llm import Base as CompletionLLM
# encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
encoder = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        return len(encoder.encode(string))
    except Exception:
        return 0

@dataclass
class CommunityReportsResult:
    """Community reports result class definition."""

    output: list[str]
    structured_output: list[dict]


class CommunityReportsExtractor:
    """Community reports extractor class definition."""

    _llm: CompletionLLM
    _extraction_prompt: str
    _output_formatter_prompt: str
    _on_error: ErrorHandlerFn
    _max_report_length: int

    def __init__(
        self,
        llm_invoker: CompletionLLM,
        extraction_prompt: Optional[str] = None,
        on_error: Optional[ErrorHandlerFn] = None,
        max_report_length: Optional[int] = None,
    ):
        """Init method definition."""
        self._llm = llm_invoker
        self._extraction_prompt = extraction_prompt or COMMUNITY_REPORT_PROMPT
        self._on_error = on_error or (lambda _e, _s, _d: None)
        self._max_report_length = max_report_length or 1500

    def __call__(self, graph: nx.Graph, callback: Optional[Callable] = None):
        communities: dict[str, dict[str, list]] = leiden.run(graph, {})
        total = sum([len(comm.items()) for _, comm in communities.items()])
        relations_df = pd.DataFrame([{"source":s, "target": t, **attr} for s, t, attr in graph.edges(data=True)])
        res_str = []
        res_dict = []
        over, token_count = 0, 0
        st = timer()
        for level, comm in communities.items():
            for cm_id, ents in comm.items():
                weight = ents["weight"]
                ents = ents["nodes"]
                ent_df = pd.DataFrame([{"entity": n, **graph.nodes[n]} for n in ents])
                rela_df = relations_df[(relations_df["source"].isin(ents)) | (relations_df["target"].isin(ents))].reset_index(drop=True)

                prompt_variables = {
                    "entity_df": ent_df.to_csv(index_label="id"),
                    "relation_df": rela_df.to_csv(index_label="id")
                }
                text = perform_variable_replacements(self._extraction_prompt, variables=prompt_variables)
                gen_conf = {"temperature": 0.3}
                try:
                    response = self._llm.chat(text, [{"role": "user", "content": "Output:"}], gen_conf)
                    token_count += num_tokens_from_string(text + response)
                    response = re.sub(r"^[^\{]*", "", response)
                    response = re.sub(r"[^\}]*$", "", response)
                    response = re.sub(r"\{\{", "{", response)
                    response = re.sub(r"\}\}", "}", response)
                    logging.debug(response)
                    response = json.loads(response)
                    if not dict_has_keys_with_types(response, [
                                ("title", str),
                                ("summary", str),
                                ("findings", list),
                                ("rating", float),
                                ("rating_explanation", str),
                            ]): continue
                    response["weight"] = weight
                    response["entities"] = ents
                except Exception as e:
                    logging.exception("CommunityReportsExtractor got exception")
                    self._on_error(e, traceback.format_exc(), None)
                    continue

                add_community_info2graph(graph, ents, response["title"])
                res_str.append(self._get_text_output(response))
                res_dict.append(response)
                over += 1
                if callback: callback(msg=f"Communities: {over}/{total}, elapsed: {timer() - st}s, used tokens: {token_count}")

        return CommunityReportsResult(
            structured_output=res_dict,
            output=res_str,
        )

    def _get_text_output(self, parsed_output: dict) -> str:
        title = parsed_output.get("title", "Report")
        summary = parsed_output.get("summary", "")
        findings = parsed_output.get("findings", [])

        def finding_summary(finding: dict):
            if isinstance(finding, str):
                return finding
            return finding.get("summary")

        def finding_explanation(finding: dict):
            if isinstance(finding, str):
                return ""
            return finding.get("explanation")

        report_sections = "\n\n".join(
            f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
        )
        return f"# {title}\n\n{summary}\n\n{report_sections}"

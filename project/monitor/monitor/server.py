from __future__ import annotations

import json
import uuid
from functools import lru_cache
from typing import Generator, cast

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from investigator.clustering import Cluster, cluster_neurons
from neurondb.filters import (
    QTILE_KEYS,
    ActivationPercentileFilter,
    AttributionFilter,
    AttributionResult,
    ComplexFilter,
    IdFilter,
    Neuron,
    NeuronDBFilter,
    NeuronPolarity,
    NeuronsMetadataDict,
    TokenFilter,
)
from neurondb.postgres import DBManager
from neurondb.view import NeuronView
from pydantic import BaseModel
from util.chat_input import make_chat_conversation
from util.errors import DBTimeoutException, EmbeddingException, LlmApiException
from util.subject import Subject, llama31_8B_instruct_config
from util.types import GenerateOutput

asgi_app = app = FastAPI()

# Add CORS middleware
asgi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Preloaded things
@lru_cache(maxsize=1)
def get_subject():
    return Subject(
        llama31_8B_instruct_config,
        output_attentions=True,
        cast_to_hf_config_dtype=True,
        nnsight_lm_kwargs={"dispatch": False},
    )


SUBJECT = get_subject()
DB = DBManager.get_instance()
PERCENTILES_PLI = NeuronView.load_percentiles(DB, SUBJECT, QTILE_KEYS)

# Session state
SESSIONS: dict[str, NeuronView] = {}
INTERVENTIONS: dict[str, list[InterventionRequest]] = {}
FILTER_CACHE: dict[str, list[Neuron]] = {}


@asgi_app.post("/register")
async def register():
    session_id = str(uuid.uuid4())
    cc = make_chat_conversation()
    nv = NeuronView(SUBJECT, DB, cc, PERCENTILES_PLI)
    SESSIONS[session_id] = nv
    return session_id


class ChatToken(BaseModel):
    token: str
    so_lens_highlight: float | None = None
    top_log_probs: list[tuple[str, float]] | None = None


class AttributionResponse(BaseModel):
    attribution_results: list[AttributionResult]
    chat_tokens: list[ChatToken]


class InterventionRequest(BaseModel):
    token_ranges: list[tuple[int, int]]
    filter: (
        NeuronDBFilter
        | ActivationPercentileFilter
        | AttributionFilter
        | TokenFilter
        | ComplexFilter
    )
    strength: float


@asgi_app.post("/register/intervention/{session_id}")
async def register_intervention(session_id: str, payload: list[InterventionRequest]):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    SESSIONS[session_id]

    intervention_id = str(uuid.uuid4())
    INTERVENTIONS[intervention_id] = payload
    return intervention_id


@asgi_app.delete("/message/clear/{session_id}")
async def clear_conversation(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    nv = SESSIONS[session_id]

    # Make new empty chat conversation
    cc = make_chat_conversation()
    nv = NeuronView(SUBJECT, DB, cc, PERCENTILES_PLI)
    SESSIONS[session_id] = nv


@asgi_app.get("/message/{session_id}")
async def send_message_sse(
    session_id: str,
    message: str | None = Query(None),
    max_new_tokens: int | None = Query(None),
    temperature: float | None = Query(None),
    intervention_id: str | None = Query(None),
):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    nv = SESSIONS[session_id]

    interventions: dict[tuple[int, int, int], float] = {}
    if intervention_id is not None:
        for intervention in INTERVENTIONS[intervention_id]:
            filter, token_ranges, strength = (
                intervention.filter,
                intervention.token_ranges,
                intervention.strength,
            )

            key = str(filter)  # Only correct if the filter doesn't depend on the message history!
            if key not in FILTER_CACHE:
                nv.set_filter(filter)
                neurons = FILTER_CACHE[key] = nv.get_neurons(with_tokens=False)
            else:
                neurons = FILTER_CACHE[key]

            neurons_metadata_dict = nv.get_neurons_metadata_dict(
                neurons, include_run_metadata=False
            )
            for neuron in neurons:
                for ts, te in token_ranges:
                    for t in range(ts, te + 1):
                        quantile = neurons_metadata_dict.general[
                            (neuron.layer, neuron.neuron)
                        ].activation_percentiles[
                            "0.9999999" if neuron.polarity == NeuronPolarity.POS else "1e-07"
                        ]
                        if quantile is not None:
                            interventions[(neuron.layer, t, neuron.neuron)] = quantile * strength

    nv.set_neuron_interventions(interventions)

    def event_generator():
        gen = cast(
            Generator[int | GenerateOutput, None, None],
            nv.send_message(
                SUBJECT,
                message,
                max_new_tokens=max_new_tokens or 64,
                temperature=temperature or 1.0,
                stream=True,
            ),
        )
        chat_tokens: list[ChatToken] = []

        for update in gen:
            if isinstance(update, int):
                chat_tokens.append(ChatToken(token=SUBJECT.decode(update)))
                data = json.dumps([ct.model_dump() for ct in chat_tokens])
                yield f"data: {data}\n\n"
            else:
                # Collate tokenwise log probs
                index_to_log_probs: dict[int, list[tuple[str, float]]] = {}
                tokenwise_log_probs = update.tokenwise_log_probs
                for i, (token_ids, log_probs) in enumerate(tokenwise_log_probs):
                    cur_log_probs = [
                        (SUBJECT.decode(token_id), log_prob)
                        for token_id, log_prob in zip(token_ids[0], log_probs[0])
                    ]
                    # output_ids_BT.shape[1] is the total sequence length, len(tokenwise_log_probs) is the number of tokens generated
                    index_to_log_probs[
                        i + update.output_ids_BT.shape[1] - len(tokenwise_log_probs)
                    ] = cur_log_probs

                # Update chat tokens with top log probs
                chat_tokens = [
                    ChatToken(**(ct.model_dump() | {"top_log_probs": index_to_log_probs.get(i)}))
                    for i, ct in enumerate(chat_tokens)
                ]
                data = json.dumps([ct.model_dump() for ct in chat_tokens])

                yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"

        nv.clear_neuron_interventions()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


class NeuronsAndMetadataResponse(BaseModel):
    neurons: list[Neuron]
    neurons_metadata_dict: NeuronsMetadataDict


@asgi_app.post("/neurons/{session_id}", response_model=NeuronsAndMetadataResponse)
async def get_neurons_with_filter(
    session_id: str,
    filter: (
        NeuronDBFilter
        | ActivationPercentileFilter
        | AttributionFilter
        | TokenFilter
        | IdFilter
        | ComplexFilter
    ) = Body(...),
):
    """
    Here the polarity matters, since it will be used for steering.
    """

    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    nv = SESSIONS[session_id]

    try:
        nv.set_filter(filter)

        # If contains an activation or attribution filter, get tokens and run metadata
        include_run_metadata = filter.contains_filter_type(
            ActivationPercentileFilter
        ) or filter.contains_filter_type(AttributionFilter)
        neurons = nv.get_neurons(
            with_tokens=include_run_metadata,
        )
        neurons_metadata_dict = nv.get_neurons_metadata_dict(neurons, include_run_metadata)

        return NeuronsAndMetadataResponse(
            neurons=neurons,
            neurons_metadata_dict=neurons_metadata_dict,
        )
    except DBTimeoutException:
        raise HTTPException(status_code=504, detail="Database timeout")
    except EmbeddingException:
        raise HTTPException(status_code=502, detail="Embedding API error")
    except LlmApiException:
        raise HTTPException(status_code=502, detail="LLM API error")


class ClusterResponse(BaseModel):
    clusters: list[Cluster]
    n_failures: int


@asgi_app.post("/neurons/cluster/{session_id}")
async def linter(
    session_id: str,
    filter: (
        NeuronDBFilter
        | ActivationPercentileFilter
        | AttributionFilter
        | TokenFilter
        | IdFilter
        | ComplexFilter
    ) = Body(...),
):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    nv = SESSIONS[session_id]
    nv.set_filter(filter)

    # Get neurons from the NeuronView
    neurons = nv.get_neurons(with_tokens=True)
    neurons_metadata_dict = nv.get_neurons_metadata_dict(neurons, include_run_metadata=True)

    # Filter neurons to keep only those with interesting descriptions
    neurons_interesting: list[Neuron] = []
    for neuron in neurons:
        neuron_metadata = neurons_metadata_dict.general.get((neuron.layer, neuron.neuron))
        if neuron_metadata is not None and neuron.polarity is not None:
            description = neuron_metadata.descriptions.get(neuron.polarity)
            if description is not None and description.is_interesting:
                neurons_interesting.append(neuron)

    clusters, n_failures = await cluster_neurons(
        neurons_interesting, neurons_metadata_dict, max_similarity_score=2, min_size=3
    )

    return ClusterResponse(clusters=clusters, n_failures=n_failures)
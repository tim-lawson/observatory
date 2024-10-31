import re
from typing import cast
from uuid import uuid4

import numpy as np
import scipy.cluster.hierarchy as sch  # type: ignore
from neurondb.filters import Neuron, NeuronPolarity, NeuronsMetadataDict
from pydantic import BaseModel, Field
from util.openai import get_openai_client_async, get_openai_embeddings_async
from util.prod_llms import get_llm_completions_async
from util.types import ChatMessage, NDIntArray

CLUSTERING_PROMPT = """You are analyzing neurons that are active on a conversation.

Provide two outputs:
1. A description of the group in about 2-5 words. Avoid using quotes or substrings. Don't include the word 'Neuron' in your description. Be VERY SPECIFIC when referencing, and I'd rather you be too specific than too general, be sure to highlight anything interesting mentioned by multiple neurons. You can comma separate it there are multiple interesting concepts.

2. A similarity score from 1 to 4 for the topical coherence of the group, where:
   1 = All neurons are about the same topic
   2 = All neurons are about topics in the same domain or general area
   3 = There are clear topical differences between a few neurons
   4 = There are clear topical differences between many neurons

Focus only on the semantic concepts when determining the similarity score.

Format your response as:
Description: [Your description here]
Similarity Score: [Your score here]
""".strip()

FS_EXAMPLES: list[ChatMessage] = [
    {
        "role": "user",
        "content": """
 - biblical citations; references "26" and "28" related to Book of Matthew
 - numeral "3" in Book of Romans; themes of sin, justification, salvation
 - specific verses (John 12, 13, 15) on Christian love, discipleship, spiritual guidance
 """.strip(),
    },
    {
        "role": "assistant",
        "content": """
Description: references to bible verses
Similarity Score: 1
""".strip(),
    },
    {
        "role": "user",
        "content": """
 - physical laws, principles, limitations, fundamental concepts in physics and nature
 - physical objects: mass, energy, movement, forces; terms like "block," "pendulum," "swing," "height"
 - forces, gravitational force, exertion, acceleration (G-forces)
""".strip(),
    },
    {
        "role": "assistant",
        "content": """
Description: gravitational forces, motion, physical laws
Similarity Score: 2
""".strip(),
    },
    {
        "role": "user",
        "content": """
 - "organic" in contexts relating to food, matter, molecules, systems, and environmental themes.
 - the abbreviation "L.A." relating to Los Angeles, often in context with societal issues, entertainment, and cultural elements.
 - proper nouns, particularly the names of famous dancers, choreographers, and locations
""".strip(),
    },
    {
        "role": "assistant",
        "content": """
Description: cultural significance of Los Angeles, waste management themes
Similarity Score: 3
""".strip(),
    },
    {
        "role": "user",
        "content": """
 - connections to the term "Alexa," such as "the" in contexts related to Alexa's features and capabilities.
 - "Alexis" activates the neuron when it follows "Alex" in contexts involving personal, relational, or narrative detail
 - contexts related to products, services, or customer experiences of "Amazon"
 - ecological or biological discussions
""".strip(),
    },
    {
        "role": "assistant",
        "content": """
Description: Alexa, Alexis, Amazon, and ecological discussions
Similarity Score: 4
""".strip(),
    },
]

# Increase this to make clusters bigger/more inclusive of varying embeddings.
SIMILARITY_THRESHOLD = 0.6


class NeuronWithDescription(Neuron):
    description: str


class Cluster(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    neurons: list[NeuronWithDescription]
    description: str  # a short description of the cluster
    similarity: int  # from 1 (most similar) to 7 (least similar)


async def cluster_neurons(
    neurons: list[Neuron],
    metadata: NeuronsMetadataDict,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    max_similarity_score: int = 2,
    min_size: int = 3,
) -> tuple[list[Cluster], int]:
    """
    Cluster a list of neurons. These should already be filtered to interesting neurons. Neurons will be automatically deduplicated.
    """

    # Remove duplicates and filter interesting neurons
    unique_neurons: list[Neuron] = []
    seen: set[tuple[int, int, NeuronPolarity | None]] = set()
    for neuron in neurons:
        key = (neuron.layer, neuron.neuron, neuron.polarity)
        if key not in seen:
            seen.add(key)
            unique_neurons.append(neuron)

    if not unique_neurons:
        print("No interesting neurons to cluster.")
        return [], 0
    elif (
        len(unique_neurons) < min_size or len(unique_neurons) < 2
    ):  # 2 is the minimum of sch.linkage
        print(f"Not enough neurons to cluster: {len(unique_neurons)}")
        return [], 0

    # Get neurons with descriptions
    neurons_with_descriptions: list[NeuronWithDescription] = []
    for n in unique_neurons:
        neuron_metadata = metadata.general.get((n.layer, n.neuron))
        assert neuron_metadata is not None
        assert n.polarity is not None
        description = neuron_metadata.descriptions[n.polarity]
        # if description is not None and description.text is not None:
        if description is not None and description.summary is not None:
            neurons_with_descriptions.append(
                NeuronWithDescription(
                    layer=n.layer,
                    neuron=n.neuron,
                    polarity=n.polarity,
                    description=description.summary,
                )
            )

    # Embed
    embeddings = await get_openai_embeddings_async(
        get_openai_client_async(),
        [n.description for n in neurons_with_descriptions],
        dimensions=3072,
    )
    embedding_array = np.array(embeddings)

    # Perform hierarchical clustering
    linkage_matrix = sch.linkage(embedding_array, method="average", metric="cosine")  # type: ignore
    cluster_labels = sch.fcluster(linkage_matrix, t=similarity_threshold, criterion="distance")  # type: ignore
    unique_labels = cast(NDIntArray, np.unique(cluster_labels))

    # Prompt a language model to generate descriptions and similarity scores
    # Final labels are the ones that we actually use
    final_labels: list[int] = []
    queries: list[str] = []
    for label in unique_labels:
        cluster_neurons = [
            n for i, n in enumerate(neurons_with_descriptions) if cluster_labels[i] == label
        ]
        if len(cluster_neurons) >= min_size:  # Filter out small clusters
            final_labels.append(label)
            queries.append("\n".join([f" - {n.description}" for n in cluster_neurons]))

    responses = await get_llm_completions_async(
        [
            [
                {
                    "role": "system",
                    "content": CLUSTERING_PROMPT,
                },
                *FS_EXAMPLES,
                {
                    "role": "user",
                    "content": query,
                },
            ]
            for query in queries
        ],
        model_category="smart",
        max_new_tokens=512,
        max_concurrency=512,
        temperature=0.5,
    )

    # Gather results
    results: list[tuple[str, int] | None] = []
    for content in responses:
        description_match = re.search(r"Description:\s*(.+)", content or "")
        score_match = re.search(r"Similarity Score:\s*(\d+)", content or "")

        if description_match and score_match:
            description = description_match.group(1).strip()
            score = int(score_match.group(1))
            results.append((description, score))
        else:
            print(f"Could not find required fields in response: {content}")
            results.append(None)  # TODO in the future retry with different sampling

    # Collect clusters
    clusters: list[Cluster] = []
    n_failures = 0
    for cluster_label, result in zip(final_labels, results):
        # Skip if we couldn't parse the response, or it didn't exist
        if result is None:
            n_failures += 1
            continue

        # Collect the data
        description, similarity = result
        cluster_data = [
            n for i, n in enumerate(neurons_with_descriptions) if cluster_labels[i] == cluster_label
        ]

        # Format the data
        neurons_in_cluster: list[NeuronWithDescription] = []
        for n in cluster_data:
            neuron = NeuronWithDescription(
                layer=n.layer, neuron=n.neuron, description=n.description
            )
            neurons_in_cluster.append(neuron)
        cluster = Cluster(
            neurons=neurons_in_cluster,
            description=description,
            similarity=similarity,
        )
        clusters.append(cluster)

    # Filter clusters
    # TODO fold this into the current function
    clusters = filter_clusters(clusters, max_similarity_score, min_size)

    return clusters, n_failures


def filter_clusters(
    clusters: list[Cluster], max_similarity: int = 3, min_size: int = 3
) -> list[Cluster]:
    """
    Filter clusters based on similarity and size, to surface the most interesting clusters.
    Then sort the filtered clusters in descending order of size.
    """
    filtered_clusters = [
        c for c in clusters if c.similarity <= max_similarity and len(c.neurons) >= min_size
    ]
    return sorted(filtered_clusters, key=lambda c: len(c.neurons), reverse=True)

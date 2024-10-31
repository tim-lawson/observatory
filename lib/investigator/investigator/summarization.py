import asyncio

import backoff
from openai import AsyncOpenAI

client = AsyncOpenAI()

SYSTEM_PROMPT = """Write a concise summary of a long neuron description in fewer than 15 words by focusing on key topics. Use abbreviations where possible and omit generic words like "the" and "of". Highlight semantic topics without mentioning "neuron" or any tokens.

# Steps

1. Identify salient topics in the description.
2. Use abbreviations to reduce phrase length.
3. Remove unnecessary articles, prepositions, and references to tokens.
4. Remove references to syntax (e.g. articles, prepositions, etc.)
5. Condense the content into a concise summary focusing on key topics.

# Output Format

- A summary in fewer than 15 words that highlights the main topics.
- Use of abbreviations where suitable.
- No mention of "neuron", "tokens" or extraneous words.
- No mention of "Describing, related, presence, refer, activates" or similar words.
- Use lowercase.

# Examples

**Input:** Political references to governance, leaders, and authority, often preceded by articles, conjunctions, and specific terms indicating the context of leadership or power dynamics.

**Output:** references to politics, governance, leaders, authority, power

**Input:** Tokens related to the United Arab Emirates, specifically variants of "UAE" and "United Arab Emirates".

**Output:** United Arab Emirates, UAE, and variants

**Input:** activation occurs on the token "Marcel" in various contexts, often relating to people, events, or works, indicating a likely reference to individuals, particularly those named Marcel.

**Output:** "Marcel" referring to people, events, or works

**Input:** presence of Greek characters or text indicating types of entities, translated words or phrases, particularly targeting Greek or Roman characters (e.g., \"\u03a1{{\u03bf}}\", \"\u03c0\u03c1\u03bf\", \"\u03bd\u03b9\u03b1{{\u03bf\u03c5}}\")

**Output:** Greek characters indicating entities; translated phrases targeting Greek or Roman characters

**Input:** Genetic or astronomical terms, particularly involving letters and numbers related to species or traits, alongside instructional scientific content.

**Output:** genetic or astronomical terms; instructional scientific content

**Input:** Tokens that refer to ""Passover,"" its rituals, and associated articles/events, often accompanied by connecting words such as ""and"" and ""for.""

**Output:** "“Passover”" and its rituals

**Input:** activation often occurs with the present tense verbs \"{{have}}\" and \"{{making}}\", emphasizing actions or states in the context.

**Output:** present tense verbs emphasizing actions or states

**Input:** Financial and technical terminology associated with blockchain mechanisms, particularly ""rewards"", and terms related to cryptocurrency system functionalities, such as ""nodes"".

**Output:** blockchain mechanisms (particularly “rewards”) and cryptocurrency terms such as “nodes”

**Input:** The neuron activates when processing specific food names, particularly Mexican dishes such as ""relleno"", ""rellen"", and Irish cultural symbols like ""leprechaun"".

**Output:** Mexican dishes such as “relleno” and Irish symbols such as “leprechaun”

**Input:** Presence of specific articles and pronouns in contexts related to popular media, such as books, movies, and cultural discussions.

**Output:** articles and pronouns related to popular media (books, movies, cultural discussions)"""

CHAT_TEMPLATE_SUMMARY = """
Write a concise summary of a long neuron description in fewer than 15 words by focusing on key topics. Use abbreviations where possible and omit generic words like "the" and "of". Highlight semantic topics without mentioning "neuron" or any tokens.

Steps:
1. Identify salient topics in the description.
2. Use abbreviations to reduce phrase length.
3. Remove unnecessary articles, prepositions, and references to tokens.
4. Remove references to syntax (e.g. articles, prepositions, etc.)
5. Condense the content into a concise summary focusing on key topics.

Description: {description}

Return the summary only.
""".strip()


async def summarize_neuron(description: str, model_name: str) -> str:
    """
    Summarize a neuron description using the system prompt.

    Example:

    'Contexts involving governance or legislative matters, particularly referencing "Sen." or related political terms, often accompanied by the prepositions "of" and "the".' -> 'governance, legislative matters, political terms (e.g., "Sen.")'

    Usage:
    ```
    summary = await summarize_neuron(description)
    ```
    """

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def make_api_call(description: str):
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": CHAT_TEMPLATE_SUMMARY.format(description=description)},
            ],
        )
        return response

    response = await make_api_call(description)

    assert response.choices[0].message.content is not None
    return response.choices[0].message.content


async def batch_summarize_neurons(
    descriptions: list[str],
    model_name: str = "gpt-4o-mini",
    concurrency: int = 10,
) -> list[str]:
    """
    Summarize a list of neurons.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def summarize_with_semaphore(description: str) -> str:
        async with semaphore:
            return await summarize_neuron(description, model_name)

    tasks = [asyncio.create_task(summarize_with_semaphore(desc)) for desc in descriptions]
    return await asyncio.gather(*tasks)

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import backoff
from anthropic import Anthropic, AsyncAnthropic
from anthropic._types import NOT_GIVEN
from anthropic.types import Message, MessageParam
from util.env import ENV
from util.types import ChatMessage


@backoff.on_exception(
    backoff.expo,
    exception=(Exception),
    max_tries=3,
    factor=2.0,
)
async def get_anthropic_chat_completion_async(
    client: AsyncAnthropic,
    messages: list[ChatMessage],
    model_name: str,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
) -> Message:
    system = messages[0]["content"] if messages[0]["role"] == "system" else None

    return await client.messages.create(
        model=model_name,
        max_tokens=max_new_tokens,
        messages=[
            MessageParam(role=m["role"], content=m["content"])
            for m in messages
            if m["role"] != "system"
        ],
        temperature=temperature,
        system=system if system is not None else NOT_GIVEN,
    )


def get_anthropic_client_sync() -> Anthropic:
    if ENV.ANTHROPIC_API_KEY is None:
        raise Exception("Missing Anthropic API key; check your .env")
    return Anthropic(api_key=ENV.ANTHROPIC_API_KEY)


def get_anthropic_client_async() -> AsyncAnthropic:
    if ENV.ANTHROPIC_API_KEY is None:
        raise Exception("Missing Anthropic API key; check your .env")
    return AsyncAnthropic(api_key=ENV.ANTHROPIC_API_KEY)


async def get_anthropic_chat_completions_async(
    client: AsyncAnthropic,
    messages_list: list[list[ChatMessage]],
    model_name: str,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    max_concurrency: int = 100,
):
    base_func = partial(
        get_anthropic_chat_completion_async,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    semaphore = asyncio.Semaphore(max_concurrency)

    async def limited_task(messages: list[ChatMessage]):
        async with semaphore:
            try:
                return await base_func(client=client, messages=messages)
            except Exception as e:  # This will catch any exception, including those from backoff
                print(f"Error in get_anthropic_chat_completion_async: {e}. Returning None.")
                return None

    tasks = [limited_task(messages) for messages in messages_list]
    responses = await asyncio.gather(*tasks)
    return responses


def parse_anthropic_completion(response: Message | None) -> str | None:
    if response is None:
        return None
    try:
        first_content = response.content[0]
        if first_content.type == "text":
            return first_content.text
        else:
            return None
    except (AttributeError, IndexError):
        return None


def get_anthropic_chat_completions_parallel(
    client: AsyncAnthropic,
    messages_list: list[list[ChatMessage]],
    model_name: str,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    max_concurrency: int = 100,
) -> list[Message | None]:
    with ThreadPoolExecutor() as executor:
        future = executor.submit(
            asyncio.run,
            get_anthropic_chat_completions_async(
                client=client,
                messages_list=messages_list,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                max_concurrency=max_concurrency,
            ),
        )
        return future.result()

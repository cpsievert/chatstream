from __future__ import annotations

__version__ = "0.1.0"

__all__ = (
    "chat_ui",
    "chat_server",
    "ChatMessageEnriched",
    "OpenAiModel",
)

import asyncio
import functools
import inspect
import json
import os
import sys
import time
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Coroutine,
    Generic,
    Literal,
    Sequence,
    TypedDict,
    TypeVar,
    cast,
)

import tiktoken
from htmltools import HTMLDependency
from openai import AsyncOpenAI
from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from .openai_types import (
    ChatCompletionStreaming,
    ChatMessage,
    OpenAiModel,
    openai_model_context_limits,
    Roles
)

if sys.version_info < (3, 10):
    from typing_extensions import ParamSpec, TypeGuard
else:
    from typing import ParamSpec, TypeGuard


client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],  # this is also the default, it can be omitted
)

DEFAULT_MODEL: OpenAiModel = "gpt-3.5-turbo"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_TEMPERATURE = 0.7
DEFAULT_THROTTLE = 0.1

# From https://icons.getbootstrap.com/icons/send/
DEFAULT_BTN_ICON = ui.HTML('<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" class="bi bi-send" viewBox="0 0 16 16"><path d="M15.854.146a.5.5 0 0 1 .11.54l-5.819 14.547a.75.75 0 0 1-1.329.124l-3.178-4.995L.643 7.184a.75.75 0 0 1 .124-1.33L15.314.037a.5.5 0 0 1 .54.11ZM6.636 10.07l2.761 4.338L14.13 2.576zm6.787-8.201L1.591 6.602l4.339 2.76z"/></svg>')


# Same as Roles
class RoleIconsType(TypedDict):
    user: ui.HTML
    assistant: ui.HTML
    system: ui.HTML


# From https://icons.getbootstrap.com/icons/
ROLE_ICONS: RoleIconsType = {
    "user": ui.HTML('<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" class="bi bi-person" viewBox="0 0 16 16"><path d="M8 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6m2-3a2 2 0 1 1-4 0 2 2 0 0 1 4 0m4 8c0 1-1 1-1 1H3s-1 0-1-1 1-4 6-4 6 3 6 4m-1-.004c-.001-.246-.154-.986-.832-1.664C11.516 10.68 10.289 10 8 10s-3.516.68-4.168 1.332c-.678.678-.83 1.418-.832 1.664z"/></svg>'),
    "assistant": ui.HTML('<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" class="bi bi-robot" viewBox="0 0 16 16"><path d="M6 12.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 0 1h-3a.5.5 0 0 1-.5-.5M3 8.062C3 6.76 4.235 5.765 5.53 5.886a26.6 26.6 0 0 0 4.94 0C11.765 5.765 13 6.76 13 8.062v1.157a.93.93 0 0 1-.765.935c-.845.147-2.34.346-4.235.346s-3.39-.2-4.235-.346A.93.93 0 0 1 3 9.219zm4.542-.827a.25.25 0 0 0-.217.068l-.92.9a25 25 0 0 1-1.871-.183.25.25 0 0 0-.068.495c.55.076 1.232.149 2.02.193a.25.25 0 0 0 .189-.071l.754-.736.847 1.71a.25.25 0 0 0 .404.062l.932-.97a25 25 0 0 0 1.922-.188.25.25 0 0 0-.068-.495c-.538.074-1.207.145-1.98.189a.25.25 0 0 0-.166.076l-.754.785-.842-1.7a.25.25 0 0 0-.182-.135"/><path d="M8.5 1.866a1 1 0 1 0-1 0V3h-2A4.5 4.5 0 0 0 1 7.5V8a1 1 0 0 0-1 1v2a1 1 0 0 0 1 1v1a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-1a1 1 0 0 0 1-1V9a1 1 0 0 0-1-1v-.5A4.5 4.5 0 0 0 10.5 3h-2zM14 7.5V13a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V7.5A3.5 3.5 0 0 1 5.5 4h5A3.5 3.5 0 0 1 14 7.5"/></svg>'),
    "system": ui.HTML('<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" class="bi bi-gear" viewBox="0 0 16 16"><path d="M8 4.754a3.246 3.246 0 1 0 0 6.492 3.246 3.246 0 0 0 0-6.492M5.754 8a2.246 2.246 0 1 1 4.492 0 2.246 2.246 0 0 1-4.492 0"/><path d="M9.796 1.343c-.527-1.79-3.065-1.79-3.592 0l-.094.319a.873.873 0 0 1-1.255.52l-.292-.16c-1.64-.892-3.433.902-2.54 2.541l.159.292a.873.873 0 0 1-.52 1.255l-.319.094c-1.79.527-1.79 3.065 0 3.592l.319.094a.873.873 0 0 1 .52 1.255l-.16.292c-.892 1.64.901 3.434 2.541 2.54l.292-.159a.873.873 0 0 1 1.255.52l.094.319c.527 1.79 3.065 1.79 3.592 0l.094-.319a.873.873 0 0 1 1.255-.52l.292.16c1.64.893 3.434-.902 2.54-2.541l-.159-.292a.873.873 0 0 1 .52-1.255l.319-.094c1.79-.527 1.79-3.065 0-3.592l-.319-.094a.873.873 0 0 1-.52-1.255l.16-.292c.893-1.64-.902-3.433-2.541-2.54l-.292.159a.873.873 0 0 1-1.255-.52zm-2.633.283c.246-.835 1.428-.835 1.674 0l.094.319a1.873 1.873 0 0 0 2.693 1.115l.291-.16c.764-.415 1.6.42 1.184 1.185l-.159.292a1.873 1.873 0 0 0 1.116 2.692l.318.094c.835.246.835 1.428 0 1.674l-.319.094a1.873 1.873 0 0 0-1.115 2.693l.16.291c.415.764-.42 1.6-1.185 1.184l-.291-.159a1.873 1.873 0 0 0-2.693 1.116l-.094.318c-.246.835-1.428.835-1.674 0l-.094-.319a1.873 1.873 0 0 0-2.692-1.115l-.292.16c-.764.415-1.6-.42-1.184-1.185l.159-.291A1.873 1.873 0 0 0 1.945 8.93l-.319-.094c-.835-.246-.835-1.428 0-1.674l.319-.094A1.873 1.873 0 0 0 3.06 4.377l-.16-.292c-.415-.764.42-1.6 1.185-1.184l.292.159a1.873 1.873 0 0 0 2.692-1.115z"/></svg>'),
}

# Make sure that the query text leaves at least this many tokens for the response. For
# example, if the model has a 4096 token limit, then the longest query will be 4096
# minus this number.
N_RESERVE_RESPONSE_TOKENS = 400

T = TypeVar("T")
P = ParamSpec("P")


class ChatMessageEnriched(TypedDict):
    """A customized version of ChatMessage, with fields for storing extra information."""

    role: Literal["system", "user", "assistant"]

    # This is the string that the user typed. (It may be run through the
    # `query_preprocessor`` to produce a different string when actually submitting the
    # query.
    content: str

    # The HTML version of the message. This is what is displayed in the chat UI. It is
    # usually the result of running `ui.markdown(content)`.
    content_html: ui.TagChild
    # Number of tokens in the message.
    token_count: int


# ================================================================================
# Chat module UI
# ================================================================================
@module.ui
def chat_ui() -> ui.Tag:
    """
    UI portion of chatstream Shiny module.
    """

    res = ui.div(
        {"class": "shiny-gpt-chat"},
        _chat_dependency(),
        ui.output_ui("session_messages_ui"),
        ui.output_ui("current_streaming_message_ui"),
        ui.output_ui("query_ui"),
    )

    return ui.fill.as_fill_item(res)

# ================================================================================
# Chat module server
# ================================================================================
# Note that most server modules are implemented as functions, but this one is
# implemented as a class, to help keep the code more organized.
@module.server
class chat_server:
    """
    Server portion of chatstream Shiny module.

    Parameters
    ----------
    model
        OpenAI model to use. Can be a string or a function that returns a string.
    api_key
        OpenAI API key to use (optional). Can be a string or a function that returns a
        string, or `None`. If `None`, then it will use the `OPENAI_API_KEY` environment
        variable for the key.
    url
        OpenAI API endpoint to use (optional). Can be a string or a function that
        returns a string, or `None`. If `None`, then it will use the default OpenAI API
        endpoint.
    system_prompt
        System prompt to use. Can be a string or a function that returns a string.
    temperature
        Temperature to use. Can be a float or a function that returns a float.
    text_input_placeholder
        Placeholder text to use for the text input. Can be a string or a function that
        returns a string, or `None` for no placeholder.
    throttle
        Throttle interval to use for incoming streaming messages. Can be a float or a
        function that returns a float.
    button_icon
        Icon to use for the button. Can be a string or a function that returns a
        string.
    button_label
        Label to use for the button. Can be a string or a function that returns a
        string.
    query_preprocessor
        Function that takes a string and returns a string. This is run on the user's
        query before it is sent to the OpenAI API. Note that is run only on the most
        recent query; previous messages in the chat history are not run through this
        function.
    answer_preprocessor
        Function that tags a string and returns a TagChild. This is run on the answer
        from the AI assistant before it is displayed in the chat UI. Note that is run on
        streaming data. As each piece of streaming data comes in, the entire accumulated
        string is run through this function.
    debug
        Whether to print debugging infromation to the console.

    Attributes
    ----------
    session_messages:
        All of the user and assistant messages in the conversation.
    hide_query_ui:
        This can be set to True to hide the query UI.
    streaming_chat_string_pieces:
        This is the current streaming chat content from the AI assistant, in the form of
        a tuple of strings, one string from each message. When not streaming, it is
        empty.
    """

    def __init__(
        self,
        input: Inputs,
        output: Outputs,
        session: Session,
        *,
        model: OpenAiModel | Callable[[], OpenAiModel] = DEFAULT_MODEL,
        api_key: str | Callable[[], str] | None = None,
        url: str | Callable[[], str] | None = None,
        system_prompt: str | Callable[[], str] = DEFAULT_SYSTEM_PROMPT,
        temperature: float | Callable[[], float] = DEFAULT_TEMPERATURE,
        text_input_placeholder: str | Callable[[], str] | None = "Your message",
        button_label: str | Callable[[], str] | None = None,
        button_icon: str | Callable[[], str] = DEFAULT_BTN_ICON,
        throttle: float | Callable[[], float] = DEFAULT_THROTTLE,
        query_preprocessor: (
            Callable[[str], str] | Callable[[str], Awaitable[str]] | None
        ) = None,
        answer_preprocessor: (
            Callable[[str], ui.TagChild]
            | Callable[[str], Awaitable[ui.TagChild]]
            | None
        ) = None,
        debug: bool = False,
    ):
        self.input = input
        self.output = output
        self.session = session

        # Ensure these are functions, even if we were passed static values.
        self.model = cast(
            # pyright needs a little help with this.
            Callable[[], OpenAiModel],
            wrap_function_nonreactive(model),
        )
        if api_key is None:
            self.api_key = get_env_var_api_key
        else:
            self.api_key = wrap_function_nonreactive(api_key)

        self.url = wrap_function_nonreactive(url)
        self.system_prompt = wrap_function_nonreactive(system_prompt)
        self.temperature = wrap_function_nonreactive(temperature)
        self.button_label = wrap_function_nonreactive(button_label)
        self.button_icon = wrap_function_nonreactive(button_icon)
        self.throttle = wrap_function_nonreactive(throttle)
        self.text_input_placeholder = wrap_function_nonreactive(text_input_placeholder)

        if query_preprocessor is None:
            query_preprocessor = lambda x: x
        self.query_preprocessor = wrap_async(query_preprocessor)

        if answer_preprocessor is None:
            # This lambda wrapper is needed to make pyright happy
            answer_preprocessor = lambda x: ui.markdown(x)
        self.answer_preprocessor = wrap_async(answer_preprocessor)

        self.print_request = debug

        # This contains a tuple of the most recent messages when a streaming response is
        # coming in. When not streaming, this is set to an empty tuple.
        self.streaming_chat_messages_batch: reactive.Value[
            tuple[ChatCompletionStreaming, ...]
        ] = reactive.Value(tuple())

        self.streaming_chat_string_pieces: reactive.Value[tuple[str, ...]] = (
            reactive.Value(tuple())
        )

        self._ask_trigger = reactive.Value(0)

        self.session_messages: reactive.Value[tuple[ChatMessageEnriched, ...]] = (
            reactive.Value(tuple())
        )

        self.hide_query_ui: reactive.Value[bool] = reactive.Value(False)

        self.reset()
        self._init_reactives()

    def reset(self) -> None:
        """
        Reset the state of this chat_server. Should not be called while streaming.
        """
        self.session_messages.set(tuple())
        self.hide_query_ui.set(False)

    def _init_reactives(self) -> None:
        """
        This method initializes the reactive components of this class.
        """

        @reactive.effect
        @reactive.event(self.streaming_chat_messages_batch)
        async def finalize_streaming_result():
            current_batch = self.streaming_chat_messages_batch()

            for message in current_batch:
                if message.choices[0].delta.content:
                    self.streaming_chat_string_pieces.set(
                        self.streaming_chat_string_pieces()
                        + (message.choices[0].delta.content,)
                    )

                finish_reason = message.choices[0].finish_reason
                if finish_reason in ["stop", "length"]:
                    # If we got here, we know that streaming_chat_string is not None.
                    current_message_str = "".join(self.streaming_chat_string_pieces())

                    if finish_reason == "length":
                        current_message_str += " [Reached token limit; Type 'continue' to continue answer.]"

                    # Update session_messages. We need to make a copy to trigger a
                    # reactive invalidation.
                    current_message: ChatMessageEnriched = {
                        "content": current_message_str,
                        "role": "assistant",
                        "content_html": await self.answer_preprocessor(
                            current_message_str
                        ),
                        "token_count": get_token_count(
                            current_message_str, self.model()
                        ),
                    }
                    self.session_messages.set(
                        self.session_messages() + (current_message,)
                    )
                    self.streaming_chat_string_pieces.set(tuple())
                    return

        @reactive.effect
        @reactive.event(self.input.ask, self._ask_trigger)
        async def perform_query():
            if self.input.query() == "":
                return

            # All previous messages, before we add the new query.
            prev_session_messages = self.session_messages()

            # First, add the current query to the session history.
            current_message: ChatMessageEnriched = {
                "content": self.input.query(),
                "role": "user",
                "content_html": ui.markdown(self.input.query()),
                "token_count": get_token_count(self.input.query(), self.model()),
            }
            self.session_messages.set(prev_session_messages + (current_message,))

            # For the query we're about to send, we need to run the current message
            # through the preprocessor.
            current_message_preprocessed: ChatMessageEnriched = current_message.copy()
            current_message_preprocessed["content"] = await self.query_preprocessor(
                current_message_preprocessed["content"]
            )
            current_message_preprocessed["token_count"] = get_token_count(
                current_message_preprocessed["content"], self.model()
            )

            # Turn it the set of messages into a list, then we'll go backward through
            # the list and keep messages until we hit the token limit.
            session_messages2 = list(prev_session_messages)
            session_messages2.append(current_message_preprocessed)

            # Count tokens, going backward.
            outgoing_messages: list[ChatMessageEnriched] = []
            tokens_total = self._system_prompt_message()["token_count"]
            max_tokens = (
                openai_model_context_limits[self.model()] - N_RESERVE_RESPONSE_TOKENS
            )
            for message in reversed(session_messages2):
                if tokens_total + message["token_count"] > max_tokens:
                    break
                else:
                    tokens_total += message["token_count"]
                    outgoing_messages.append(message)

            outgoing_messages.append(self._system_prompt_message())
            outgoing_messages.reverse()

            outgoing_messages_normalized = chat_messages_enriched_to_chat_messages(
                outgoing_messages
            )

            if self.print_request:
                print(json.dumps(outgoing_messages_normalized, indent=2))
                print(f"TOKENS USED: {tokens_total}")

            extra_kwargs = {}
            if self.url() is not None:
                extra_kwargs["url"] = self.url()

            # Launch a Task that updates the chat string asynchronously. We run this in
            # a separate task so that the data can come in without need to await it in
            # this Task (which would block other computation to happen, like running
            # reactive stuff).
            messages: StreamResult[ChatCompletionStreaming] = stream_to_reactive(
                client.chat.completions.create(  # pyright: ignore
                    model=self.model(),
                    messages=outgoing_messages_normalized,  # pyright: ignore
                    stream=True,
                    temperature=self.temperature(),
                    **extra_kwargs,
                ),
                throttle=self.throttle(),
            )

            # Set this to a non-empty tuple (with a blank string), to indicate that
            # streaming is happening.
            self.streaming_chat_string_pieces.set(("",))

            @reactive.effect
            def copy_messages_to_batch():
                self.streaming_chat_messages_batch.set(messages())

        @render.ui
        def session_messages_ui():
            messages_html: list[ui.Tag] = []
            for message in self.session_messages():
                if message["role"] == "system":
                    # Don't show system messages.
                    continue

                messages_html.append(
                    message_html(message["content_html"], message["role"])
                )

            return messages_html

        @render.ui
        async def current_streaming_message_ui():
            pieces = self.streaming_chat_string_pieces()

            # Only display this content while streaming. Once the streaming is done,
            # this content will disappear and an identical-looking one will be added to
            # the `session_messages_ui` output.
            if len(pieces) == 0:
                return None

            content = "".join(pieces)
            if content == "":
                content = "\u2026"  # zero-width string
            else:
                content = await self.answer_preprocessor(content)

            return message_html(content, "assistant")

        # Note: assets/chat.js makes an assumption that we invalidate the query UI when
        # the chat starts streaming.
        @render.ui
        @reactive.event(self.hide_query_ui, self.streaming_chat_string_pieces)
        def query_ui():
            if self.hide_query_ui():
                return None

            disabled = len(self.streaming_chat_string_pieces()) > 0

            return ui.div(
                {"class": "query-container"},
                ui.div(
                    {"class": "input-group"},
                    # input_text_area() adds an additional container that makes it harder to style.
                    # but shiny.js will bind to any <textarea>, so just use a plain textarea.
                    ui.tags.textarea(
                        id=module.resolve_id("query"),
                        class_="form-control",
                        style="resize:none;",
                        placeholder="Responding..." if disabled else self.text_input_placeholder(),
                        disabled=disabled,
                    ),
                    ui.input_action_button(
                        "ask",
                        label=self.button_label(),
                        icon=self.button_icon(),
                        disabled=disabled,
                        class_="btn-primary",
                    ),
                )
            )

    def _system_prompt_message(self) -> ChatMessageEnriched:
        return {
            "role": "system",
            "content": self.system_prompt(),
            "content_html": "",
            "token_count": get_token_count(self.system_prompt(), self.model()),
        }

    async def _delayed_set_query(self, query: str, delay: float) -> None:
        await asyncio.sleep(delay)
        async with reactive.lock():
            ui.update_text_area("query", value=query, session=self.session)
            await reactive.flush()

        # Short delay before triggering ask_trigger.
        safe_create_task(self._delayed_new_query_trigger(0.2))

    async def _delayed_new_query_trigger(self, delay: float) -> None:
        await asyncio.sleep(delay)
        async with reactive.lock():
            self._ask_trigger.set(self._ask_trigger() + 1)
            await reactive.flush()

    def ask(self, query: str, delay: float = 1) -> None:
        """Programmatically ask a question."""
        safe_create_task(self._delayed_set_query(query, delay))


# ==============================================================================
# Helper functions
# ==============================================================================


def get_env_var_api_key() -> str:
    import os

    key = os.environ.get("OPENAI_API_KEY")
    if key is None:
        raise ValueError(
            "Please set the OPENAI_API_KEY environment variable to your OpenAI API key."
        )

    return key


def get_token_count(s: str, model: OpenAiModel) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(s))


def message_html(content: ui.TagChild, role: Roles) -> ui.Tag:
    icon = ROLE_ICONS[role]
    return ui.div(
        {"class": role + "-message message-container"},
        ui.span(
            {"class": "badge rounded-pill text-bg-secondary"},
            {"style": "height: fit-content;"},
            icon
        ),
        ui.div(
            {"class": "message-content"},
            content,
        ),
    )


# A place to keep references to Tasks so they don't get GC'd prematurely, as directed in
# asyncio.create_task docs
running_tasks: set[asyncio.Task[Any]] = set()


def safe_create_task(task: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
    t = asyncio.create_task(task)
    running_tasks.add(t)
    t.add_done_callback(running_tasks.remove)
    return t


class StreamResult(Generic[T]):
    _read: Callable[[], tuple[T, ...]]
    _cancel: Callable[[], bool]

    def __init__(self, read: Callable[[], tuple[T, ...]], cancel: Callable[[], bool]):
        self._read = read
        self._cancel = cancel

    def __call__(self) -> tuple[T, ...]:
        """
        Perform a reactive read of the stream. You'll get the latest value, and you will
        receive an invalidation if a new value becomes available.
        """

        return self._read()

    def cancel(self) -> bool:
        """
        Stop the underlying stream from being consumed. Returns False if the task is
        already done or cancelled.
        """
        return self._cancel()


# Converts an async generator of type T into a reactive source of type tuple[T, ...].
def stream_to_reactive(
    func: AsyncGenerator[T, None] | Awaitable[AsyncGenerator[T, None]],
    throttle: float = 0,
) -> StreamResult[T]:
    val: reactive.Value[tuple[T, ...]] = reactive.Value(tuple())

    async def task_main():
        nonlocal func
        if inspect.isawaitable(func):
            func = await func  # type: ignore
        func = cast(AsyncGenerator[T, None], func)

        last_message_time = time.time()
        message_batch: list[T] = []

        async for message in func:
            # This print will display every message coming from the server.
            # print(json.dumps(message, indent=2))
            message_batch.append(message)

            if time.time() - last_message_time > throttle:
                async with reactive.lock():
                    val.set(tuple(message_batch))
                    await reactive.flush()

                last_message_time = time.time()
                message_batch = []

        # Once the stream has ended, flush the remaining messages.
        if len(message_batch) > 0:
            async with reactive.lock():
                val.set(tuple(message_batch))
                await reactive.flush()

    task = safe_create_task(task_main())

    return StreamResult(val.get, lambda: task.cancel())


def chat_message_enriched_to_chat_message(
    msg: ChatMessageEnriched,
) -> ChatMessage:
    return {"role": msg["role"], "content": msg["content"]}


def chat_messages_enriched_to_chat_messages(
    messages: Sequence[ChatMessageEnriched],
) -> list[ChatMessage]:
    return list(chat_message_enriched_to_chat_message(msg) for msg in messages)


def is_async_callable(
    obj: Callable[P, T] | Callable[P, Awaitable[T]]
) -> TypeGuard[Callable[P, Awaitable[T]]]:
    """
    Returns True if `obj` is an `async def` function, or if it's an object with a
    `__call__` method which is an `async def` function. This function should generally
    be used in this code base instead of iscoroutinefunction().
    """
    if inspect.iscoroutinefunction(obj):
        return True
    if hasattr(obj, "__call__"):  # noqa: B004
        if inspect.iscoroutinefunction(obj.__call__):  # type: ignore
            return True

    return False


def wrap_async(
    fn: Callable[P, T] | Callable[P, Awaitable[T]]
) -> Callable[P, Awaitable[T]]:
    """
    Given a synchronous function that returns T, return an async function that wraps the
    original function. If the input function is already async, then return it unchanged.
    """

    if is_async_callable(fn):
        return fn

    fn = cast(Callable[P, T], fn)

    @functools.wraps(fn)
    async def fn_async(*args: P.args, **kwargs: P.kwargs) -> T:
        return fn(*args, **kwargs)

    return fn_async


def wrap_function_nonreactive(x: T | Callable[[], T]) -> Callable[[], T]:
    """
    This function is used to normalize three types of things so they are wrapped in the
    same kind of object:

    - A value
    - A non-reactive function that return a value
    - A reactive function that returns a value

    All of these will be wrapped in a non-reactive function that returns a value.
    """
    if not callable(x):
        return lambda: x

    @functools.wraps(x)
    def fn_nonreactive() -> T:
        with reactive.isolate():
            return x()

    return fn_nonreactive


def _chat_dependency():
    return HTMLDependency(
        "shiny-gpt-chat",
        __version__,
        source={"package": "chatstream", "subdir": "assets"},
        script={"src": "chat.js"},
        stylesheet={"href": "chat.css"},
    )

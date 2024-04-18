from __future__ import annotations

from shiny import App, Inputs, Outputs, Session, reactive, ui

import chatstream

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select(
            "model",
            "Model",
            choices=["gpt-3.5-turbo", "gpt-4"],
            selected="gpt-3.5-turbo",
        ),
        ui.input_slider(
            "temperature",
            ui.tooltip(
                "Temperature",
                "Lower values are more deterministic. Higher values are more random and unpredictable."
            ),
            min=0,
            max=2,
            value=0.7,
            step=0.05,
        ),
        ui.input_switch("auto_converse", "Auto-conversation", value=True),
        ui.input_slider(
            "auto_converse_delay",
            "Conversation delay (seconds)",
            min=0,
            max=3,
            value=2.4,
            step=0.2,
        ),
        ui.hr(class_="mt-auto"),
        ui.p(
            "Built with ",
            ui.a("Shiny for Python", href="https://shiny.rstudio.com/py/"),
        ),
        ui.p(
            ui.a(
                "Source code",
                href="https://github.com/wch/chatstream",
                target="_blank",
            ),
        ),
        title="Shiny ChatGPT",
        position="right",
        style="height:100%;"
    ),
    ui.layout_columns(
        ui.card(chatstream.chat_ui("chat1")),
        ui.card(chatstream.chat_ui("chat2")),
    ),
    window_title="Shiny ChatGPT",
    fillable=True
)

# ======================================================================================


def server(input: Inputs, output: Outputs, session: Session):
    chat_session1 = chatstream.chat_server(
        "chat1",
        model=input.model,
        temperature=input.temperature,
    )
    chat_session2 = chatstream.chat_server(
        "chat2",
        model=input.model,
        temperature=input.temperature,
    )

    # Which chat module has the most recent completed response from the server.
    most_recent_module = reactive.Value(0)

    @reactive.Effect
    @reactive.event(chat_session1.session_messages)
    def _():
        with reactive.isolate():
            if not input.auto_converse() or most_recent_module() == 1:
                return

        # Don't try to converse if there are no messages.
        if len(chat_session1.session_messages()) == 0:
            return

        last_message = chat_session1.session_messages()[-1]
        if last_message["role"] == "assistant":
            chat_session2.ask(
                last_message["content"],
                input.auto_converse_delay(),
            )
            most_recent_module.set(1)

    @reactive.Effect
    @reactive.event(chat_session2.session_messages)
    def _():
        with reactive.isolate():
            if not input.auto_converse() or most_recent_module() == 2:
                return

        # Don't try to converse if there are no messages.
        if len(chat_session2.session_messages()) == 0:
            return

        last_message = chat_session2.session_messages()[-1]
        if last_message["role"] == "assistant":
            chat_session1.ask(
                last_message["content"],
                input.auto_converse_delay(),
            )
            most_recent_module.set(2)


app = App(app_ui, server)

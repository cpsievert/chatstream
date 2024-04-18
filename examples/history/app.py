from shiny import App, Inputs, Outputs, Session, ui

import chatstream

app_ui = ui.page_sidebar(
    ui.sidebar(
        title="Chat history",
    ),
    chatstream.chat_ui("mychat"),
    fillable=True,
    fillable_mobile=True,
)


def server(input: Inputs, output: Outputs, session: Session):
    chatstream.chat_server("mychat")


app = App(app_ui, server)

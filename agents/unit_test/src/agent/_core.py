from typing import TypedDict

import polars as pl

class ChatState(TypedDict):
    messages: list


class RunTimeState(TypedDict):
    df: pl.DataFrame
from typing import Literal, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, Tool

from ._core import RunTimeState

def build_tools(runtime: RunTimeState):
    """
    NOTE: This is a horrible hack that I used because nothing else would work, and I am really ashamed of it.
    However, I could not find any other way to effectively pass runtime variables such as DataFrames to my tools.

    I tried using RunnableConfig(https://python.langchain.com/docs/concepts/tools/), but ultimately I was running in so many issues 
    when passing it to the various langgraph components that I decided it was not worth it.
    """

    @tool
    def just_chat():
        """When no appropriate tool should be used, just chat with the user."""

        # NOTE: This "dummy" tool was added because LLama3.1 really likes to use tools 
        # when they are available, even if they are not necessary.

        return "Just chat with the user."



    @tool
    def get_df_property(prop: Literal["len", "unique", "value_counts"], column: Optional[str]=None):
        """Get property from the DataFrame."""

        match prop:
            case "len":
                return f"len(df): {len(runtime['df'])}"
            
            case "unique":
                if column is None:
                    return f"ERROR: column must be specified to get property {prop}."
                
                return f"df[{column}].unique(): {runtime['df'][column].unique()}"
                

            case "unique":
                if column is None:
                    return f"ERROR: column must be specified to get property {prop}."f"column must be specified to get property {prop}."
                
                return f"df[{column}].value_counts(): {runtime['df'][column].value_counts()}"


    return just_chat, get_df_property



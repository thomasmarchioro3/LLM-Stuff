from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

from ._core import ChatState


def build_graph(llm: Runnable, tools: list[BaseTool]):

    tool_node = ToolNode(tools=tools)

    def llm_node(state: ChatState, config: RunnableConfig | None = None) -> ChatState:
        response = llm.invoke(state["messages"], config=config)

        state = ChatState(
            messages=state["messages"] + [response],
        )
        return state

    def tools_node(state: ChatState, config: RunnableConfig | None = None) -> ChatState:

        tool_output = tool_node.invoke(state, config=config)

        state = ChatState(
            messages=state["messages"] + tool_output["messages"],
        )

        return state

    def router(state: ChatState) -> str:
        msg = state["messages"][-1]
        if hasattr(msg, "tool_calls"):
            if len(msg.tool_calls) > 0:
                return "tools"
        return "end"

    builder = StateGraph(ChatState, RunnableConfig)
    builder.add_node("llm", llm_node)
    builder.add_node("tools", tools_node)

    """
    Structure:

    START -> llm -> router (tools?) -- YES --> tools --> JUMP llm (loop)
                                    -- NO ---> END
    """
    builder.add_edge(START, "llm")
    builder.add_edge("tools", "llm")
    builder.add_conditional_edges("llm", router, {"tools": "tools", "end": END})

    graph = builder.compile()

    return graph

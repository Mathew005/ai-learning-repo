from typing import List, Any
from rich.tree import Tree
from rich.markup import escape
from langchain_core.runnables import RunnableLambda
from app.langchain_lab.core import console

def render_state_tree(step_name: str, state: dict, new_keys: List[str] = None):
    """
    Helper to draw a pretty tree of the current dictionary state.
    Useful for visualizing LCEL chain progressions.
    """
    tree = Tree(f"[bold blue]{step_name}[/bold blue]")
    
    # Sort keys to keep display stable
    for key in sorted(state.keys()):
        is_new = key in (new_keys or [])
        
        # Format Key
        key_style = "bold green" if is_new else "bold white"
        prefix = "➕ " if is_new else "  "
        
        # Format Value (Truncate nicely)
        val = str(state[key]).replace("\n", " ") # Single line for cleanliness
        val_display = (val[:80] + "...") if len(val) > 80 else val
        
        # Escape to prevent rich markup errors
        val_display = escape(val_display)
        
        val_style = "green" if is_new else "dim cyan"
        
        tree.add(f"{prefix}[{key_style}]{key}[/{key_style}]: [{val_style}]{val_display}[/{val_style}]")
        
    console.print(tree)
    console.print("")

def visualize_step(step_name: str, new_keys: List[str] = None):
    """
    Returns a RunnableLambda that logs the state without modifying it.
    Use this inside an LCEL chain to debug/visualize steps.
    
    Usage:
        chain = Step1 | visualize_step("After Step 1") | Step 2
    """
    def _log(state: Any):
        if isinstance(state, dict):
            render_state_tree(step_name, state, new_keys)
        else:
            # Fallback for non-dict state
            console.print(f"[bold blue]{step_name}[/bold blue]: {escape(str(state))}")
        return state # Pass through unchanged
    
    return RunnableLambda(_log)

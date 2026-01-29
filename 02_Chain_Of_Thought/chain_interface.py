import streamlit as st
from shared.llm import LLMEngine
import traceback

def render():
    st.header("Phase 2: Chain of Thought")
    st.caption("Sequential reasoning with two models.")

    # --- Sidebar for this module ---
    with st.sidebar:
        st.subheader("Chain Configuration")
        
        slots = LLMEngine.get_model_slots()
        slot_options = list(slots.keys())
        
        # Model 1 Selection
        st.markdown("### Step 1 Model")
        slot1_name = st.selectbox("Select Model 1", slot_options, key="ph2_slot1", index=0)
        model_1 = slots[slot1_name]
        st.caption(f"Using: `{model_1}`")
        
        # Model 2 Selection
        st.markdown("### Step 2 Model")
        slot2_name = st.selectbox("Select Model 2", slot_options, key="ph2_slot2", index=1 if len(slot_options) > 1 else 0)
        model_2 = slots[slot2_name]
        st.caption(f"Using: `{model_2}`")

        st.divider()
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, key="ph2_temp")
        
        if st.button("Clear Chain", key="ph2_clear"):
            st.session_state.ph2_messages = []
            st.rerun()

    # --- System Prompts ---
    col1, col2 = st.columns(2)
    with col1:
        sys_prompt_1 = st.text_area("System Prompt 1 (Reasoning)", value="You are a helpful assistant. Think step-by-step.", height=100, key="ph2_sys1")
    with col2:
        sys_prompt_2 = st.text_area("System Prompt 2 (Final Answer)", value="You are a helpful assistant. Synthesize the information.", height=100, key="ph2_sys2")

    # --- Chat Logic ---
    if "ph2_messages" not in st.session_state:
        st.session_state.ph2_messages = []

    # Display History
    for msg in st.session_state.ph2_messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif role == "assistant_step1":
            with st.expander("Chain of Thought (Model 1)", expanded=False):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(content)

    # Input and Generation
    if prompt := st.chat_input("Ask something..."):
        # 1. Add User Message to History
        st.session_state.ph2_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Process with Model 1 (Chain of Thought)
        step1_response_text = ""
        with st.chat_message("assistant"): # Placeholder for visual consistency or could use a status container
            status_container = st.status("Thinking...", expanded=True)
            
            try:
                # Prepare Model 1 Context
                # We interpret previous history? For simplicity in "Chain of Thought" typically strictly sequential single-turn or limited context
                # But here lets try to maintain some history consistency if possible, OR just treat this as a fresh chain for the new prompt.
                # Let's simplify: pass current prompt to Model 1 with System Prompt 1.
                
                status_container.write("Step 1: Reasoning with " + slot1_name)
                
                msgs_1 = [{"role": "system", "content": sys_prompt_1}, {"role": "user", "content": prompt}]
                # Note: If we wanted full history, we'd need to parse `ph2_messages` to filter what goes to Model 1. 
                # For now, let's keep it focused on the current turn's chain.
                
                response_1 = LLMEngine.chat(model_1, msgs_1, temperature)
                step1_response_text = response_1 if response_1 else "No response."
                
                status_container.markdown(f"**Step 1 Output:**\n{step1_response_text}")
                
                # Store Step 1
                st.session_state.ph2_messages.append({"role": "assistant_step1", "content": step1_response_text})

                # 3. Process with Model 2 (Final Answer)
                status_container.write("Step 2: Synthesizing with " + slot2_name)
                
                # Context for Model 2: System 2 -> User Prompt -> Model 1 Response (as context or 'assistant' turn?)
                # A good pattern: System -> User -> (User says: Here is preliminary thinking) -> ...
                # Or: System -> User (Prompt) -> Assistant (Step 1 Output) ... wait, if we do that, Model 2 might think IT generated step 1.
                # Better: System 2, User: <Prompt> \n\n <Context/Thoughts>: ...
                
                final_input = f"Original Query: {prompt}\n\nPre-computation/Reasoning:\n{step1_response_text}"
                msgs_2 = [{"role": "system", "content": sys_prompt_2}, {"role": "user", "content": final_input}]
                
                response_2 = LLMEngine.chat(model_2, msgs_2, temperature)
                final_response_text = response_2 if response_2 else "No response."
                
                status_container.update(label="Complete", state="complete", expanded=False)
                
                st.markdown(final_response_text)
                st.session_state.ph2_messages.append({"role": "assistant", "content": final_response_text})

            except Exception as e:
                status_container.update(label="Error", state="error")
                error_msg = f"Chain Error: {type(e).__name__}: {e}"
                st.error(error_msg)
                traceback.print_exc()

import streamlit as st
from shared.llm import LLMEngine
import traceback
import time

def render():
    st.header("Phase 2: Chain of Thought")
    st.caption("Sequential reasoning with a dynamic chain of models.")

    # --- Sidebar: Chain Configuration ---
    with st.sidebar:
        st.subheader("Chain Topology")
        
        # Topology Control: Number Input
        # We bind this directly to a key in session state, but we need to initialize it first if not present
        if "ph2_num_steps" not in st.session_state:
            st.session_state.ph2_num_steps = 2
            
        num_steps = st.number_input(
            "Number of Steps", 
            min_value=2, 
            max_value=10, 
            value=st.session_state.ph2_num_steps,
            step=1,
            key="_ph2_num_steps_input"
        )
        
        # Sync the input with our persistent state variable if it changes
        if num_steps != st.session_state.ph2_num_steps:
             st.session_state.ph2_num_steps = num_steps
             st.rerun()

        st.divider()
        
        # Model Selection with Expanders
        slots = LLMEngine.get_model_slots()
        slot_options = list(slots.keys())
        selected_models = []
        
        for i in range(st.session_state.ph2_num_steps):
            step_num = i + 1
            default_idx = 0 if i % 2 == 0 else (1 if len(slot_options) > 1 else 0)
            
            # Flat structure
            st.markdown(f"**Step {step_num} Model**")
            slot_name = st.selectbox(
                f"Select Model", 
                slot_options, 
                key=f"ph2_slot_{i}", 
                index=default_idx,
                label_visibility="collapsed"
            )
            model_ref = slots[slot_name]
            st.caption(f"Using: `{model_ref}`")
            selected_models.append((slot_name, model_ref))
            st.divider()

        st.divider()
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, key="ph2_temp")
        
        if st.button("Clear Chain", key="ph2_clear"):
            st.session_state.ph2_messages = []
            st.rerun()

    # --- Main Area: System Prompts ---
    
    status_placeholders = []
    
    for i in range(st.session_state.ph2_num_steps):
        step_num = i + 1
        
        # 1. Status Banner Placeholder (e.g. "Running...")
        # We put this ABOVE the system prompt
        ph = st.empty()
        status_placeholders.append(ph)
        
        # 2. System Prompt Input (Static)
        label = f"Step {step_num}: System Prompt"
        key = f"ph2_sys_{i}"
        
        # Default Logic
        default_val = "You are a helpful assistant. Think step-by-step." if i < st.session_state.ph2_num_steps - 1 else "You are a helpful assistant. Synthesize the final answer."
        
        st.text_area(
            label,
            value=st.session_state.get(key, default_val),
            height=100,
            key=key
        )

    st.divider()

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
        elif role.startswith("assistant_step"):
            step_num = role.split("_")[-1]
            with st.expander(f"Chain Step {step_num}", expanded=False):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(content)

    # Input and Execution
    if prompt := st.chat_input("Start the Chain..."):
        # 1. User Message
        st.session_state.ph2_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Sequential Execution
        current_input_context = prompt
        accumulated_chain = "" 
        
        try:
            for i in range(st.session_state.ph2_num_steps):
                step_id = i + 1
                model_display_name, model_id = selected_models[i]
                sys_prompt = st.session_state.get(f"ph2_sys_{i}", "")
                
                # --- START MARKER ---
                status_placeholders[i].warning(f"▶️ Running Step {step_id} with {model_display_name}...", icon="⏳")
                
                # Prepare Messages
                full_context_msg = ""
                if i == 0:
                    full_context_msg = current_input_context
                else:
                    full_context_msg = f"Original Query: {prompt}\n\nPrevious Reasoning/Context:\n{accumulated_chain}"
                
                msgs = [
                    {"role": "system", "content": sys_prompt}, 
                    {"role": "user", "content": full_context_msg}
                ]
                
                # Run Inference
                response_text = LLMEngine.chat(model_id, msgs, temperature)
                
                if not response_text or "Error" in response_text:
                     raise Exception(f"Model {model_display_name} returned error/empty: {response_text}")

                # --- END MARKER ---
                status_placeholders[i].success(f"Step {step_id} Complete", icon="✅")
                
                accumulated_chain += f"\n--- Step {step_id} Output ---\n{response_text}\n"
                
                is_final = (i == st.session_state.ph2_num_steps - 1)
                
                if not is_final:
                    st.session_state.ph2_messages.append({
                        "role": f"assistant_step{step_id}", 
                        "content": response_text
                    })
                    with st.expander(f"Step {step_id} Result", expanded=True):
                        st.markdown(response_text)
                else:
                    status_placeholders[i].empty() # Clear final checkmark for aesthetics? Or keep it. Let's keep it brief then clear.
                    time.sleep(0.5) 
                    status_placeholders[i].empty() 
                    
                    st.session_state.ph2_messages.append({
                        "role": "assistant",
                        "content": response_text
                    })
                    with st.chat_message("assistant"):
                        st.markdown(response_text)
                        
                # Brief pause
                time.sleep(0.2)
                # Cleanup the 'running' banner to 'Done' or clear
                if not is_final:
                     time.sleep(1.0)
                     status_placeholders[i].empty() 

        except Exception as e:
            # --- ERROR HANDLING ---
            status_placeholders[i].error(f"Failed: {e}", icon="❌")
            st.error(f"🛑 Execution Halted at Step {step_id}")
            traceback.print_exc()
            return # HALT

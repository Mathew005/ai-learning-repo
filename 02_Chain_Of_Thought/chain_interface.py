import streamlit as st
from shared.llm import LLMEngine
import traceback
import time

def render():
    st.header("Phase 2: Chain of Thought")
    st.caption("Sequential reasoning with a dynamic chain of models.")

    # --- Session State for Dynamic Steps ---
    if "ph2_num_steps" not in st.session_state:
        st.session_state.ph2_num_steps = 2

    # --- Sidebar: Chain Configuration ---
    with st.sidebar:
        st.subheader("Chain Topology")
        
        # Step Management
        col_add, col_rem = st.columns(2)
        if col_add.button("➕ Add Step"):
            st.session_state.ph2_num_steps += 1
        
        if col_rem.button("➖ Remove Step"):
            if st.session_state.ph2_num_steps > 2:
                st.session_state.ph2_num_steps -= 1
            else:
                st.toast("Minimum 2 steps required.", icon="⚠️")

        st.divider()
        
        # Model Selection for Each Step
        slots = LLMEngine.get_model_slots()
        slot_options = list(slots.keys())
        
        selected_models = []
        
        for i in range(st.session_state.ph2_num_steps):
            st.markdown(f"**Step {i+1} Model**")
            # Default indices: Flip flop between 0 and 1 for variety if available
            default_idx = 0 if i % 2 == 0 else (1 if len(slot_options) > 1 else 0)
            
            slot_name = st.selectbox(
                f"Select Model for Step {i+1}", 
                slot_options, 
                key=f"ph2_slot_{i}", 
                index=default_idx,
                label_visibility="collapsed"
            )
            model_ref = slots[slot_name]
            selected_models.append((slot_name, model_ref))
            st.caption(f"Using: `{model_ref}`")
            st.divider()

        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, key="ph2_temp")
        
        if st.button("Clear Chain", key="ph2_clear"):
            st.session_state.ph2_messages = []
            st.rerun()

    # --- Main Area: System Prompts with Status Placeholders ---
    
    # We need to capture the placeholders to update them during execution
    status_placeholders = []
    system_prompts = []
    
    # Render System Prompt Inputs
    # We'll use a loop. To make it look nice, maybe columns if few, or vertical list.
    # Vertical list is safer for N steps.
    
    for i in range(st.session_state.ph2_num_steps):
        # Header with Placeholder for "Active" status
        col_head, col_status = st.columns([0.8, 0.2])
        with col_head:
            st.subheader(f"Step {i+1}: System Prompt")
        with col_status:
            # This empty container will be populated during execution
            ph = st.empty()
            status_placeholders.append(ph)
            
        # Default prompt logic
        default_prompt = "You are a helpful assistant. Think step-by-step." if i < st.session_state.ph2_num_steps - 1 else "You are a helpful assistant. Synthesize the final answer."
        
        sys_p = st.text_area(
            "Prompt Content", 
            value=default_prompt, 
            height=100, 
            key=f"ph2_sys_{i}",
            label_visibility="collapsed"
        )
        system_prompts.append(sys_p)

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
            # Intermediate steps
            step_num = role.split("_")[-1] # extraction might be 'step1', 'step2' etc
            with st.expander(f"Chain Step {step_num}", expanded=False):
                st.markdown(content)
        elif role == "assistant":
            # Final step
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
        
        # Container for execution updates
        with st.chat_message("assistant"):
            overall_status = st.status("Initializing Chain...", expanded=True)
            
            try:
                for i in range(st.session_state.ph2_num_steps):
                    step_id = i + 1
                    model_display_name, model_id = selected_models[i]
                    sys_prompt = system_prompts[i]
                    
                    # A. Highlight the System Prompt in UI
                    status_placeholders[i].info("Wait... ⏳") # Indicate pending?
                    status_placeholders[i].warning(f"**Running {model_display_name}...** 🏃‍♂️")
                    
                    overall_status.write(f"**Step {step_id}**: Using {model_display_name}...")
                    
                    # B. Prepare Messages
                    # Context strategy: 
                    # Step 1: System + User Prompt
                    # Step N: System + (User Prompt + Previous Chain)
                    
                    full_context_msg = ""
                    if i == 0:
                        full_context_msg = current_input_context
                    else:
                        full_context_msg = f"Original Query: {prompt}\n\nPrevious Reasoning/Context:\n{accumulated_chain}"
                    
                    msgs = [
                        {"role": "system", "content": sys_prompt}, 
                        {"role": "user", "content": full_context_msg}
                    ]
                    
                    # C. Run Inference
                    response_text = LLMEngine.chat(model_id, msgs, temperature)
                    if not response_text:
                        response_text = "No response generated."

                    # D. Update UI & History
                    accumulated_chain += f"\n--- Step {step_id} Output ---\n{response_text}\n"
                    
                    # Store intermediate steps as 'assistant_step{i}'
                    # Identify if this is the FINAL step
                    is_final = (i == st.session_state.ph2_num_steps - 1)
                    
                    if not is_final:
                         st.session_state.ph2_messages.append({
                             "role": f"assistant_step{step_id}", 
                             "content": response_text
                         })
                         overall_status.write(f"Step {step_id} Complete. ✅")
                         with st.expander(f"Step {step_id} Result"):
                             st.markdown(response_text)
                    else:
                        # Final Result
                        overall_status.update(label="Chain Complete!", state="complete", expanded=False)
                        st.session_state.ph2_messages.append({
                            "role": "assistant",
                            "content": response_text
                        })
                        st.markdown(response_text)
                    
                    # E. Clear Highlight
                    status_placeholders[i].empty() # Remove highlight
                    status_placeholders[i].success(f"Done ✅")
                    time.sleep(0.5) # Brief pause to see the success state

            except Exception as e:
                overall_status.update(label="Chain Error", state="error")
                st.error(f"Error at Step {step_id}: {e}")
                traceback.print_exc()
                # Clear active highlights on error
                for ph in status_placeholders:
                    ph.empty()

import streamlit as st
from shared.llm import LLMEngine

def render():
    st.header("Phase 1: Chat Interface")
    st.caption("Demonstrating Basic LLM Chat")

    # --- Sidebar for this module ---
    with st.sidebar:
        st.subheader("Chat Configuration")
        # Select Model Slot
        slots = LLMEngine.get_model_slots()
        # Create a list of "Slot Name: Model Name" for display
        slot_options = list(slots.keys())
        selected_slot_name = st.selectbox("Select Model Slot", slot_options, key="ph1_slot")
        
        selected_model = slots[selected_slot_name]
        
        st.caption(f"Using: `{selected_model}`")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, key="ph1_temp")
        
        if st.button("Clear Chat", key="ph1_clear"):
            st.session_state.ph1_messages = []
            st.rerun()

    # --- Chat Logic ---
    if "ph1_messages" not in st.session_state:
        st.session_state.ph1_messages = []

    # Display History
    for msg in st.session_state.ph1_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input and Generation
    if prompt := st.chat_input("Ask something..."):
        st.session_state.ph1_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Prepare messages
                messages_for_llm = [{"role": m["role"], "content": m["content"]} for m in st.session_state.ph1_messages]
                
                try:
                    response = LLMEngine.chat(selected_model, messages_for_llm, temperature)
                    full_response = response if response else "No response received."
                except Exception as e:
                    import traceback
                    error_msg = f"Chat Error: {type(e).__name__}: {e}"
                    print(f"[CHAT_INTERFACE] {error_msg}")
                    traceback.print_exc()
                    full_response = error_msg
                
                st.markdown(full_response)
        
        st.session_state.ph1_messages.append({"role": "assistant", "content": full_response})

import streamlit as st

# List of options
options = ["A", "B", "C", "D"]

# Initialize session state to track the selected items
if "selected" not in st.session_state:
    st.session_state.selected = []

# Function to determine available options based on selection
def get_available_options(selected):
    if "D" in selected:
        # If "D" is selected, disable "A", "B", and "C"
        return [option for option in options if option == "D"]
    else:
        return options

# Get the available options based on the current selection
available_options = get_available_options(st.session_state.selected)

# Display multiselect widget with dynamic options
st.session_state.selected = st.multiselect(
    "Select your options:",
    available_options,
    default=st.session_state.selected
)

# Display the selected items
st.write("You selected:", st.session_state.selected)

import re

# Read the file
with open('streamlit_app.py', 'r') as file:
    lines = file.readlines()

# Find the section that displays consultant matches
for i, line in enumerate(lines):
    if 'st.write("### Consultant Matches")' in line and i+1 < len(lines) and 'response = getattr' in lines[i+1]:
        # Replace the next line with our new display code
        new_display_code = [
            '                                \n',
            '                                # Get the consultant matches from the final state\n',
            '                                matches = getattr(final_state, \'consultant_matches\', [])\n',
            '                                st.session_state.current_matches = matches\n',
            '                                \n',
            '                                if matches:\n',
            '                                    st.write("🎯 **Best Matching Consultants**")\n',
            '                                    for i, consultant in enumerate(matches, 1):\n',
            '                                        with st.expander(f"👨‍💼 Consultant {i}: {consultant[\'Name\']}"):\n',
            '                                            cols = st.columns(2)\n',
            '                                            with cols[0]:\n',
            '                                                st.markdown(f"**👤 Age:** {consultant[\'Age\']}")\n',
            '                                                st.markdown(f"**🎓 Education:** {consultant[\'Education\']}")\n',
            '                                                st.markdown(f"**💼 Industry Expertise:** {consultant[\'Industry Expertise\']}")\n',
            '                                            with cols[1]:\n',
            '                                                st.markdown(f"**📅 Availability:** {\'Yes\' if consultant[\'Availability\'] else \'No\'}")\n',
            '                                                st.markdown(f"**📝 Bio:** {consultant[\'Bio\']}")\n',
            '                                            \n',
            '                                            st.markdown("---")\n',
            '                                            st.markdown("**🔍 Match Analysis:**")\n',
            '                                            st.markdown(consultant[\'Match Analysis\'])\n',
            '                                else:\n',
            '                                    st.error("😔 No matching consultants found.")\n',
            '                                    \n',
            '                                # Also display the AI-generated response if available\n',
            '                                response = getattr(final_state, \'response\', "")\n',
            '                                if response:\n',
            '                                    with st.expander("💡 AI Analysis of Matches"):\n',
            '                                        st.write(response)\n'
        ]
        
        # Replace the next two lines with our new display code
        lines[i+1:i+3] = new_display_code
        print(f"Updated display code at line {i+1}")
        break

# Find the second section that displays consultant matches
for i, line in enumerate(lines):
    if 'st.write("### Consultant Matches")' in line and i+1 < len(lines) and 'st.write(state.response if state.response else "No consultant matches found.")' in lines[i+1]:
        # Replace the next line with our new display code
        new_display_code = [
            '                \n',
            '                # Get the consultant matches from the state\n',
            '                matches = getattr(state, \'consultant_matches\', [])\n',
            '                st.session_state.current_matches = matches\n',
            '                \n',
            '                if matches:\n',
            '                    st.write("🎯 **Best Matching Consultants**")\n',
            '                    for i, consultant in enumerate(matches, 1):\n',
            '                        with st.expander(f"👨‍💼 Consultant {i}: {consultant[\'Name\']}"):\n',
            '                            cols = st.columns(2)\n',
            '                            with cols[0]:\n',
            '                                st.markdown(f"**👤 Age:** {consultant[\'Age\']}")\n',
            '                                st.markdown(f"**🎓 Education:** {consultant[\'Education\']}")\n',
            '                                st.markdown(f"**💼 Industry Expertise:** {consultant[\'Industry Expertise\']}")\n',
            '                            with cols[1]:\n',
            '                                st.markdown(f"**📅 Availability:** {\'Yes\' if consultant[\'Availability\'] else \'No\'}")\n',
            '                                st.markdown(f"**📝 Bio:** {consultant[\'Bio\']}")\n',
            '                            \n',
            '                            st.markdown("---")\n',
            '                            st.markdown("**🔍 Match Analysis:**")\n',
            '                            st.markdown(consultant[\'Match Analysis\'])\n',
            '                else:\n',
            '                    st.error("😔 No matching consultants found.")\n',
            '                    \n',
            '                # Also display the AI-generated response if available\n',
            '                response = getattr(state, \'response\', "")\n',
            '                if response:\n',
            '                    with st.expander("💡 AI Analysis of Matches"):\n',
            '                        st.write(response)\n'
        ]
        
        # Replace the next line with our new display code
        lines[i+1:i+2] = new_display_code
        print(f"Updated second display code at line {i+1}")
        break

# Write the updated content back to the file
with open('streamlit_app.py', 'w') as file:
    file.writelines(lines)

print("Updated streamlit_app.py to display consultant matches in a better format")

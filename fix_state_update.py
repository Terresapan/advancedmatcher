# Read the file
with open('streamlit_app.py', 'r') as file:
    lines = file.readlines()

# Find the section that creates the state_update
for i, line in enumerate(lines):
    if '# Create a new state with the updated requirements' in line:
        # Find the end of the state_update section
        for j in range(i, len(lines)):
            if 'state_update = {' in lines[j]:
                # Find the end of the state_update dictionary
                for k in range(j, len(lines)):
                    if '}' in lines[k]:
                        # Replace the state_update section with our new code
                        new_state_update_code = [
                            '                                # Create a new state with the updated requirements\n',
                            '                                # We need to preserve the original state structure but update specific fields\n',
                            '                                # Always include project_summary and project_text to ensure they\'re not lost\n',
                            '                                original_summary = st.session_state.get(\'project_summary\', "")\n',
                            '                                original_text = st.session_state.get(\'project_text\', "")\n',
                            '                                \n',
                            '                                state_update = {\n',
                            '                                    "analysis": updated_requirements,\n',
                            '                                    "requirements_approved": True,\n',
                            '                                    "project_summary": original_summary,\n',
                            '                                    "project_text": original_text\n',
                            '                                }\n'
                        ]
                        
                        # Replace the state_update section
                        lines[i:k+1] = new_state_update_code
                        print(f"Updated state_update section at line {i}")
                        break
                break
        break

# Find the section that adds additional context
for i, line in enumerate(lines):
    if '# Update both project_summary and project_text to include additional context' in line:
        # Find the end of the additional context section
        for j in range(i, len(lines)):
            if 'print(f"Added additional context: {additional_text}")' in lines[j]:
                # Replace the additional context section with our new code
                new_additional_context_code = [
                    '                                    # Update both project_summary and project_text to include additional context\n',
                    '                                    state_update["project_summary"] = state_update["project_summary"] + f"\\n\\nAdditional Context: {additional_text}"\n',
                    '                                    state_update["project_text"] = state_update["project_text"] + f"\\n\\nAdditional Context: {additional_text}"\n',
                    '                                    \n',
                    '                                    print(f"Added additional context: {additional_text}")\n'
                ]
                
                # Replace the additional context section
                lines[i:j+1] = new_additional_context_code
                print(f"Updated additional context section at line {i}")
                break
        break

# Write the updated content back to the file
with open('streamlit_app.py', 'w') as file:
    file.writelines(lines)

print("Updated streamlit_app.py to always include project_summary in state_update")

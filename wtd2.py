

import streamlit as st
import os

def save_uploaded_file(uploadedfile):
  directory = os.getcwd()  # Get the current working directory

  with open(os.path.join(directory,uploadedfile.name),"wb") as f:
     f.write(uploadedfile.getbuffer())
  return st.success("Saved file :{} in tempDir".format(uploadedfile.name))


def main():
    st.title("Upload and Save XLSX File")

    # Allow user to upload a file
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])
    # st.write(uploaded_file)
    
    # Specify the desired save location

    # Create the desired save location directory if it doesn't exist
    if uploaded_file:
        # file_name = st.write(uploaded_file.name)
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
        # ---------------
        # save_directory = "test"
        # os.makedirs(save_directory, exist_ok=True)
        # Process and save the uploaded file to the desired location
        # process_and_save_file(uploaded_file, save_directory)
        # -----------------
        
        save_uploaded_file(uploaded_file)

        
        
if __name__ == "__main__":
    main()

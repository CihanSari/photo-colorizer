from colorizer import ImageColorizer
import streamlit as st
import numpy as np
import cv2
import os

# Streamlit app layout
st.title('Photo Colorizer')

# Sidebar for folder selection
st.sidebar.header('Folder Selection')
folder = st.sidebar.text_input('Folder', '')

if st.sidebar.button('Load Folder'):
    if os.path.isdir(folder):
        img_types = (".png", ".jpg", "jpeg", ".tiff", ".bmp")
        fnames = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(img_types)]
        st.session_state['fnames'] = fnames
    else:
        st.sidebar.error('Invalid folder path')

if 'fnames' in st.session_state:
    fnames = st.session_state['fnames']
else:
    fnames = []

convert_to_gray = st.sidebar.checkbox('Convert to gray first')

# Main section for displaying images and actions
if fnames:
    cols = st.columns(2)
    col1, col2 = cols

    # Process and display all images
    processed_images = []
    colorizer = ImageColorizer(clip_limit=2.0, tile_grid_size=(8, 8))
    for fname in fnames:
        file_path = os.path.join(folder, fname)
        image = cv2.imread(file_path)
        
        col1.image(image, caption=f'Original: {fname}', use_column_width=True)

        if convert_to_gray:
            gray_3_channels = colorizer.convert_to_grayscale(image)
            image, colorized = colorizer.colorize_image(cv2_frame=gray_3_channels)
        else:
            _, colorized = colorizer.colorize_image(image_filename=file_path)
        colorized_image = cv2.imencode('.png', colorized)[1].tobytes()
        col2.image(colorized_image, caption=f'Colorized: {fname}', use_column_width=True)
        processed_images.append((fname, colorized))

    # Save options
    save_individual = st.sidebar.selectbox('Save individual images', ['Select an image'] + fnames)
    if save_individual != 'Select an image':
        save_path = st.text_input(f'Enter save path for {save_individual}')
        if st.button(f'Save {save_individual}'):
            if save_path:
                colorized_image = [img for name, img in processed_images if name == save_individual][0]
                cv2.imwrite(save_path, colorized_image)
                st.success(f'Image {save_individual} saved successfully!')
            else:
                st.error('Please enter a valid save path')

    save_all_path = "out"
    if st.button('Save All Images'):
        for fname, colorized in processed_images:
            save_fname = os.path.join(save_all_path, fname)
            cv2.imwrite(save_fname, colorized)
        st.success('All images saved successfully!')
else:
    st.write('No images loaded')

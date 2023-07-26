import streamlit as st
from generate import gen_by_vec

st.set_page_config(page_title='Tumors Generator', initial_sidebar_state="auto", menu_items=None)

# @st.cache_data
with st.sidebar:

    tumor_type = st.select_slider(
        'Select tumor type',
        options=['Benign', 'Malignant']
    )

    shape = st.select_slider(
        'Select a the shape',
        options=['ellipse', 'heart']
    )

    margin = st.select_slider(
        'Select margin type',
        options=['Microlobulated', 'heart']
    )


    center_x = st.slider('center_x', min_value=0.0, max_value=100.0, step=0.01)
    center_y = st.slider('center_y', min_value=0.0, max_value=100.0, step=0.01)

    radius_a = st.slider('radius_a', min_value=0.0, max_value=100.0, step=0.01)
    radius_b = st.slider('radius_b', min_value=0.0, max_value=100.0, step=0.01)

    angel = st.slider('angel', min_value=0.0, max_value=100.0, step=0.01)
    density = st.slider('density', min_value=0.0, max_value=10.0, step=0.01)
    edge_r = st.slider('edge_r', min_value=0.0, max_value=10.0, step=0.01)
    edge_num = st.slider('edge_num', min_value=0.0, max_value=10.0, step=0.01)

    shadow = st.select_slider(
        'Select shadow type',
        options=['black', 'write']
    )
    spot_num = st.slider('spot_num', min_value=0.0, max_value=10.0, step=0.01)

feature_dict = {

    "tumor_type": {
        'Benign': 0,
        'Malignant': 1
    },
    "shape": {
        "ellipse": 0,
    },
    "margin": {
        "Microlobulated": 0,
    },
    "shadow": {
        "black": 0,
        'write': 0,
    }
}


col1, col2 = st.columns(2)

vector = [feature_dict['tumor_type'][tumor_type], feature_dict['shape'][shape], feature_dict['margin'][margin], center_x, center_y, radius_a, radius_b, angel, density, edge_r, edge_num, feature_dict['shadow'][shadow], spot_num]
pic = gen_by_vec(vector)

col1.write(f"""# Feature Vector
{vector}
""")

col2.write('# Generated image')
col2.image(pic)


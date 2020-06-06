import sys

sys.path.append("./src/")

import streamlit as st
from models.seq2seq import Seq2seq
from models.transformer import transformer
from utils import load_model, epoch_time, get_max_version, get_torch_device
from inference.inference_helpers import GreedyDecoder, BeamDecoder
import torch
import time


def load_st_model(name, version):
    state_dict, inpLang, optLang, hp = load_model(name=name, version=version)
    if name == "seq2seq":
        model = Seq2seq(**hp)
    elif name == "transformer":
        src_pad_idx = inpLang.word2idx[inpLang.special["pad_token"]]
        trg_pad_idx = optLang.word2idx[optLang.special["pad_token"]]
        model = transformer(src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx, **hp)
    model.load_state_dict(state_dict)
    return model, inpLang, optLang, hp


st.sidebar.title("Question Generation Models")

model_name = st.sidebar.selectbox("Model", ["seq2seq", "transformer"])
version = get_max_version(model_name)
version_selected = st.sidebar.slider(
    "Select version", min_value=min(version), max_value=max(version)
)
model, inpLang, optLang, hp = load_st_model(model_name, version_selected)

if hp["squad"] == False:
    st.markdown("## Translation Model")
else:
    st.markdown("## Question Generation Model")


dec_type = st.radio("Decoding Type", ("Greedy", "Beam"))
max_length = st.slider("Max Decoding Length", 2, 100)

if dec_type == "Greedy":
    decoder = GreedyDecoder(model=model, inpLang=inpLang, optLang=optLang)
elif dec_type == "Beam":
    decoder = BeamDecoder(model=model, inpLang=inpLang, optLang=optLang)
    beam_size = st.slider("Beam width", 1, 15)


decoder.to(get_torch_device())

st.markdown("# Input")
txt = st.text_area("Context", "john used to live in canada")

st.markdown("# Output")

if dec_type == "Greedy":
    st.markdown(" ".join(decoder.greedy(txt, max_len=max_length)))
elif dec_type == "Beam":
    outputs = decoder.beam(txt, max_len=max_length, beam_width=beam_size)
    for idx, i in enumerate(outputs):
        st.markdown(f"**{str(idx)}**.   " + " ".join(i[1]))
        st.markdown(f" \t Score : ```{i[0]}```")

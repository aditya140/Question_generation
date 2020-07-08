import sys

sys.path.append("./src/")

import streamlit as st
from models.seq2seq import Seq2seq
from models.transformer import transformer
from models.attn_seq2seq import AttnSeq2seq
from utils import load_model, epoch_time, get_max_version, get_torch_device,load_test_df
from inference.inference_helpers import GreedyDecoder, BeamDecoder
import torch
import time
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from scipy.special import softmax



def load_st_model(name, version):
    state_dict, inpLang, optLang, hp = load_model(name=name, version=version)
    if name == "seq2seq":
        model = Seq2seq(**hp)
    elif name == "transformer":
        src_pad_idx = inpLang.word2idx[inpLang.special["pad_token"]]
        trg_pad_idx = optLang.word2idx[optLang.special["pad_token"]]
        model = transformer(src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx, **hp)
    elif name == "attn_seq2seq":
        model = AttnSeq2seq(**hp)
    model.load_state_dict(state_dict)
    df = load_test_df(name=name,version=version)
    return model, inpLang, optLang, hp, df


def plot_heatmap(inp,opt,mask):
    mask = mask.detach().cpu()
    mask = mask[:,:len(inp)+1].numpy().transpose(1,0)
    # mask = mask / mask.max(axis=0)
    mask=softmax(mask)
    ax = sns.heatmap(mask, linewidth=0.5,xticklabels=opt,yticklabels=["<SOS>"] + inp,cmap="Blues")
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    return ax



st.sidebar.title("Question Generation Models")

model_name = st.sidebar.selectbox("Model", ["seq2seq", "transformer","attn_seq2seq"])
version = get_max_version(model_name)
if len(version)==0:
    version_selected=-2
else:
    version_selected = st.sidebar.slider(
            "Select version", min_value=min(version)-1, max_value=max(version)
        )
if len(version)>0 and version_selected!=-1:
    model, inpLang, optLang, hp, test_df = load_st_model(model_name, version_selected)
    if hp["squad"] == False:
        st.markdown("## Translation Model")
    else:
        st.markdown("## Question Generation Model")


    dec_type = st.radio("Decoding Type", ("Greedy", "Beam"))
    max_length = st.slider("Max Decoding Length", 2, 100)
    show_test_df = st.button("Test Results") 
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
        sent , att_map = decoder.greedy(txt, max_len=max_length)
        inp_str,opt_str,attention_mask = att_map
        if attention_mask!=None:
            plot_heatmap(inp_str,opt_str,attention_mask.squeeze(1))
        st.markdown(" ".join(sent))
        if attention_mask!=None:
            st.pyplot()
    elif dec_type == "Beam":
        outputs = decoder.beam(txt, max_len=max_length, beam_width=beam_size)
        for idx, i in enumerate(outputs):
            st.markdown(f"**{str(idx)}**.   " + " ".join(i[1]))
            st.markdown(f" \t Score : ```{i[0]}```")

    if show_test_df:
        offset=st.number_input("Offset",min_value=0,max_value=int(test_df.shape[0]/50),step=50,value=0)

        st.markdown("### Test Results")
        st.table(test_df.iloc[offset:offset+50])

elif version_selected==-1:
    st.markdown("Select Valid Model")
else:
    st.markdown("No Models available")

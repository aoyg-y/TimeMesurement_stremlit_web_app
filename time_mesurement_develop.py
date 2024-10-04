import streamlit as st
import matplotlib.pyplot as plt
import tempfile
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import math

class TimeMesurement:
    def __init__(self,filepath):
        #投げられたfile pathから動画と総フレーム、FPSを取得
        self.video = cv2.VideoCapture(filepath)
        self.frame_num = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    def lightness(self):
        v_means = []
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.video.read()
        while ret:
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            #動画の中心付近のv値の平均をv_meansに追加
            v_means.append(v[:, int(self.width/2) - 5: int(self.width/2) + 5].mean())
            ret, frame = self.video.read()
        
        return np.arange(len(v_means)), np.array(v_means)
    
    def valid_convolve(self, xx, size):
            #補正された移動平均を出力
            b = np.ones(size)/size
            xx_mean = np.convolve(xx, b, mode="same")

            n_conv = math.ceil(size/2)

            # 補正部分
            xx_mean[0] *= size/n_conv
            for i in range(1, n_conv):
                xx_mean[i] *= size/(i+n_conv)
                xx_mean[-i] *= size/(i + n_conv - (size % 2)) 
            # size%2は奇数偶数での違いに対応するため
            return xx_mean
    

    def peakpeak(self,threshold=4,size=10):
        #輝度の時間変化
        x = self.lightness()[1]
        #時間変化の移動平均を産出
        x_mean = self.valid_convolve(x, size) 
        #移動平均との差から鋭いピークを取得
        x_resd = x - x_mean

        #thresholdが大きすぎてすべて0にされてしまうとerrorが出るのでピークが出てくるまでthresholdを下げる
        while True:
            try:
                #threshold以下の増減はすべて0に
                x_flatten = np.array(list(map(lambda x: 0 if x<threshold else x, x_resd)))
                x_fla_df = pd.DataFrame(x_flatten,columns=["frame"])
                non_zero_frames =  x_fla_df[x_fla_df["frame"] != 0 ].index
                #peak_frames = self.sequence_average(non_zero_frames)
                break
            except IndexError:
                threshold -= 1

        return non_zero_frames
    

    
def fileload_measure(file_path):
    #openCVに投げるにはfileのpathのstrオブジェクトが必要なので、アップロードされたファイルを一時ファイルとして保存してからpathを取得している。
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(file_path.read())

    TM = TimeMesurement(temp_file.name)
    #一時ファイルは閉じておく
    temp_file.close()

    peaks = TM.peakpeak()
    return TM,peaks

def peak_select(st_file_path):
    #動画がアップロードされていて、かつ動画解析をしていないとき
    if (st_file_path is not None) and ("TM" not in st.session_state):
        TM, peaks = fileload_measure(st_file_path)
        st.session_state.TM = TM
        st.session_state.peaks = peaks

    #動画解析ができているとき
    if "TM" in st.session_state:
        # セッションに現在のフレーム位置とピークリストのインデックスを保持
        if 'peak_idx' not in st.session_state:
            st.session_state.peak_idx = 0  # peakリストのインデックス
        if 'frame_pos' not in st.session_state:
            st.session_state.frame_pos = st.session_state.peaks[st.session_state.peak_idx]
        if "selected_frame" not in st.session_state:
            st.session_state.selected_frame = []


        # ボタンで操作を選択
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            if st.button("5つ前のフレーム"):
                st.session_state.frame_pos = max(0,st.session_state.frame_pos - 5)
        with col2:
            if st.button("1つ前のフレーム"):
                st.session_state.frame_pos = max(0,st.session_state.frame_pos - 1)
        with col3:
            if st.button("これにする"):
                st.session_state.selected_frame.append(st.session_state.frame_pos)
                st.session_state.peak_idx = min(len(st.session_state.peaks) - 1, st.session_state.peak_idx + 1)
                st.session_state.frame_pos = st.session_state.peaks[st.session_state.peak_idx]
        with col4:
            if st.button("1つ次のフレーム"):
                st.session_state.frame_pos = min(st.session_state.TM.frame_num-1, st.session_state.frame_pos + 1)
        with col5:
            if st.button("5つ次のフレーム"):
                st.session_state.frame_pos = min(st.session_state.TM.frame_num-1, st.session_state.frame_pos + 5)
        with col6:
            if st.button("選択せず次へ"):
                st.session_state.peak_idx = min(len(st.session_state.peaks) - 1, st.session_state.peak_idx + 1)
                st.session_state.frame_pos = st.session_state.peaks[st.session_state.peak_idx]

        # 現在のフレームに移動
        st.session_state.TM.video.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_pos)
        ret, frame = st.session_state.TM.video.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            st.image(img_pil, caption=f"Frame {st.session_state.frame_pos}")
            st.text(st.session_state.selected_frame)
        
        return st.session_state.selected_frame

def change_movie(button_name):
    st.button(button_name, on_click=change_page)

def change_page():
    st.session_state.page_control += 1
 
st.title("区間タイム計測くん")
st.subheader("使い方")
st.text('''''')
st.text(list(st.session_state.keys()))

if "page_control" in st.session_state and st.session_state.page_control == 1:
    if "clear_TM" in st.session_state:
        for ss in list(st.session_state.keys()):
            if ss not in ["s_frames","page_control"]:
                del st.session_state[ss]
    st.subheader("動画のアップロード")
    st_file_path = st.file_uploader("ゴール地点の動画をアップロードしてください")
    st.session_state.g_frames = peak_select(st_file_path)
    change_movie("測定結果を表示")

elif "page_control" in st.session_state and st.session_state.page_control == 2:
    result = []
    for s,g in zip(st.session_state.s_frames,st.session_state.g_frames):
        result.append(g-s)
    st.text(result)

else:
    st.session_state.page_control = 0
    st.subheader("動画のアップロード")
    st_file_path = st.file_uploader("スタート地点の動画をアップロードしてください")
    st.session_state.s_frames = peak_select(st_file_path)
    st.session_state.clear_TM = True
    change_movie("ゴール地点の動画を選択")
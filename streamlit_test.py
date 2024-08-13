import streamlit as st
import matplotlib.pyplot as plt
import tempfile
import numpy as np
import pandas as pd
import cv2
import math

class TimeMesurement:
    def __init__(self,filepath):
        #投げられたfile pathから動画と総フレーム、FPSを取得
        self.video = cv2.VideoCapture(filepath)
        self.frame_num = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    def lightness(self):
        frames = []
        v_means = []
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(1,self.frame_num):
            ret, frame = self.video.read()
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            #動画の中心付近のv値の平均をv_meansに追加
            v_means.append(v[:, int(self.width/2) - 5: int(self.width/2) + 5].mean())
            frames.append(i)
        return np.array(frames), np.array(v_means)
    
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
    
    def sequence_average(self, x):
        #[34,35,56,67,89,90]のような形から、連続している部分の中心を出力(ピークトップとする)
        averages = []
        group = [x[0]]
        for i in range(1,len(x)):
            if x[i] == x[i-1] + 1:
                group.append(x[i])
            
            else:
                averages.append(sum(group) / len(group))
                group = [x[i]]
        averages.append(sum(group) / len(group))
        return averages

    def peakpeak(self,size=10,threshold=4):
        #輝度の時間変化
        x = self.lightness()[1]
        #時間変化の移動平均を産出
        x_mean = self.valid_convolve(x, size) 
        #移動平均との差から鋭いピークを取得
        x_resd = x - x_mean
        #threshold以下の増減はすべて0に
        x_flatten = np.array(list(map(lambda x: 0 if x<threshold else x, x_resd)))
        x_fla_df = pd.DataFrame(x_flatten)
        non_zero_frames =  x_fla_df[x_fla_df[x_fla_df.index[0]] != 0 ].index
        peak_frames = self.sequence_average(non_zero_frames)
        return np.array(peak_frames)
    

    
def fileload_measure(file_path):
    #openCVに投げるにはfileのpathのstrオブジェクトが必要なので、アップロードされたファイルを一時ファイルとして保存してからpathを取得している。
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(file_path.read())

    TM = TimeMesurement(temp_file.name)
    #一時ファイルは閉じておく
    temp_file.close()
    #dataframe型に変換して数値を丸めた
    time = pd.DataFrame(TM.peakpeak() / TM.fps).round(2)
    time.columns = ["time (s)"]
    #fig, ax = plt.subplots()
    #ax.plot(*TM.lightness())
    #st.pyplot(fig)
    return time

st.title("区間タイム計測くん")
st.subheader("使い方")
st.text('''
1.動画の撮影
スタート地点/ゴール地点の２つの動画を撮影
注1: ２つの動画の撮影開始が同時になるようにしてください(iOSアプリの「SVCam」が便利そうです)
注2： 動画の中心とスタート/ゴールラインをきっちり合わせてください。
注3: 人が通過するとすべて検知されるので気を付けてください。 (通過する順番が交差しなければ問題はないので、周回している人がいる際は同じ向きで抜かさないよう走るといいと思います)
        
2.動画のアップロード
下の欄から動画をスタート、ゴール地点の順にアップロードしてください。(アップロードするまではerror表示が出てますが無視してください)
通過した時刻が表示されます。２つともアップロードされて、通過が検知された人数が同じならタイムが通過順に表示されます。
もし、通過した人数と検知された人数が異なる事象を発見されましたら動画とともにご連絡ください。機能改善のための参考とさせていただきます。
        ''')
st.subheader("動画のアップロード")

st_file_path = st.file_uploader("スタート地点の動画をアップロードしてください")
st_time = fileload_measure(st_file_path)
st.text("スタート地点通過")
st.dataframe(st_time)

go_file_path = st.file_uploader("ゴール地点の動画をアップロードしてください")
go_time = fileload_measure(go_file_path)
st.text(f"ゴール地点通過(秒)")
st.dataframe(go_time)

st.subheader("解析結果")
if len(st_time) != len(go_time):
    st.text("ゴールとスタートを通過した人数が違います")
else:
    st.dataframe(go_time - st_time)


#動画とグラフをその場で確認できるようにしたいね。
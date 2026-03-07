import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tempfile

# =============================
# 解析関数（元コードそのまま）
# =============================

def run_analysis(csv_path, output_root, r_2, d, graph_limits):

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join(output_root, base_name)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path, encoding="shift_jis")

    calibration = {
        1:(1.00052,0.02530),
        2:(1.00301,-0.21838),
        3:(0.99915,0.06390),
        4:(0.99734,0.12859),
        5:(1.00725,-0.17171),
        6:(0.99723,0.08716),
        7:(0.99941,-0.04587),
        8:(0.99614,0.12996),
    }

    for i,(a,b) in calibration.items():
        df.iloc[:,i] = df.iloc[:,i]*a + b

    x1=np.array([1,2,3,4])
    x2=np.array([5+d/10,6+d/10,7+d/10,8+d/10])

    r=r_2/2
    A=np.pi*r**2
    rA=0.01/A
    K=398

    results=[]

    for i in range(len(df)):
        try:

            y1=df.iloc[i,1:5].astype(float).values
            y2=df.iloc[i,5:9].astype(float).values

            if np.isnan(y1).any() or np.isnan(y2).any():
                continue

            y1s=np.sort(y1)[::-1]
            y2s=np.sort(y2)[::-1]

            a1,b1=np.polyfit(x1,y1s,1)
            a2,b2=np.polyfit(x2,y2s,1)

            min_y1=4.5*a1+b1
            max_y2=(5+d/10-0.5)*a2+b2

            dt=min_y1-max_y2
            q=-K*a2*rA
            R=dt/q*100

            results.append([i+1,a1,b1,a2,b2,dt,q,R])

        except:
            pass

    df_result=pd.DataFrame(results,columns=[
        "行番号","a1","b1","a2","b2","dt","q","R"
    ])

    df_result["R_average"]=df_result["R"][::-1].rolling(
        window=10,min_periods=1
    ).mean()[::-1]

    time=df.iloc[:,0]

    # =============================
    # 温度グラフ 上
    # =============================

    fig1=plt.figure(figsize=(8,5))

    for i,col in enumerate(df.columns[1:5],start=2):
        plt.scatter(time,df[col],s=10,label=f"CH{i}")

    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [°C]")
    plt.title("Temperature (CH2–CH5)")
    plt.legend()
    plt.xlim(graph_limits["temp_upper"]["xmin"],graph_limits["temp_upper"]["xmax"])
    plt.ylim(graph_limits["temp_upper"]["ymin"],graph_limits["temp_upper"]["ymax"])
    plt.tight_layout()

    path1=os.path.join(output_dir,"CH2_5_upper_temp.png")
    plt.savefig(path1,dpi=300)
    plt.close()

    # =============================
    # 温度グラフ 下
    # =============================

    fig2=plt.figure(figsize=(8,5))

    for i,col in enumerate(df.columns[5:9],start=7):
        plt.scatter(time,df[col],s=10,label=f"CH{i}")

    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [°C]")
    plt.title("Temperature (CH7–CH10)")
    plt.legend()
    plt.xlim(graph_limits["temp_lower"]["xmin"],graph_limits["temp_lower"]["xmax"])
    plt.ylim(graph_limits["temp_lower"]["ymin"],graph_limits["temp_lower"]["ymax"])
    plt.tight_layout()

    path2=os.path.join(output_dir,"CH7_10_lower_temp.png")
    plt.savefig(path2,dpi=300)
    plt.close()

    # =============================
    # Rグラフ
    # =============================

    fig3=plt.figure(figsize=(7,5))

    plt.scatter(df_result["行番号"],df_result["R"],s=10,label="R",color="red")
    plt.scatter(df_result["行番号"],df_result["R_average"],s=10,label="R_average",color="blue")

    plt.xlabel("Time [s]")
    plt.ylabel("Thermal Resistance")
    plt.legend()
    plt.tight_layout()

    path3=os.path.join(output_dir,"R_plot.png")
    plt.savefig(path3,dpi=300)
    plt.close()

    # =============================
    # CSV保存
    # =============================

    result_path=os.path.join(output_dir,"result.csv")
    df_result.to_csv(result_path,index=False,encoding="shift_jis")

    return path1,path2,path3,result_path


# =============================
# Streamlit UI
# =============================

st.title("熱抵抗解析 Webアプリ")

uploaded_file=st.file_uploader("CSVファイル")

r2=st.number_input("r2 [cm]",value=0.8)
d=st.number_input("d [mm]",value=1.0)

st.subheader("グラフ設定")

temp_xmin=st.number_input("時間 xmin",value=0)
temp_xmax=st.number_input("時間 xmax",value=8000)

temp1_ymin=st.number_input("上側温度 ymin",value=20)
temp1_ymax=st.number_input("上側温度 ymax",value=110)

temp2_ymin=st.number_input("下側温度 ymin",value=20)
temp2_ymax=st.number_input("下側温度 ymax",value=30)

if st.button("解析開始"):

    if uploaded_file is None:
        st.error("CSVをアップロードしてください")
    else:

        tmp_dir=tempfile.mkdtemp()

        csv_path=os.path.join(tmp_dir,uploaded_file.name)

        with open(csv_path,"wb") as f:
            f.write(uploaded_file.getbuffer())

        graph_limits={

        "temp_upper":{
        "xmin":temp_xmin,
        "xmax":temp_xmax,
        "ymin":temp1_ymin,
        "ymax":temp1_ymax
        },

        "temp_lower":{
        "xmin":temp_xmin,
        "xmax":temp_xmax,
        "ymin":temp2_ymin,
        "ymax":temp2_ymax
        }

        }

        path1,path2,path3,result_path=run_analysis(
            csv_path,tmp_dir,r2,d,graph_limits
        )

        st.success("解析完了")

        st.image(path1)
        st.image(path2)
        st.image(path3)

        st.download_button(
            "result.csvダウンロード",
            open(result_path,"rb"),
            file_name="result.csv"
        )
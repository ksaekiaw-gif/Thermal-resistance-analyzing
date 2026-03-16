import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tempfile
import zipfile
import io

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

    # 測定終了時間
    end_time = time.iloc[-1]

    # -------------------------
    # 終了1000s前までの平均
    # -------------------------

    mask_1000 = time >= (end_time - 1000)

    R_avg_1000 = df_result.loc[mask_1000, "R"].mean()

    # -------------------------
    # 終了2000s前までの平均
    # -------------------------

    mask_2000 = time >= (end_time - 2000)

    R_avg_2000 = df_result.loc[mask_2000, "R"].mean()
    
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
    plt.ylabel("Thermal Resistance R [mm$^2$ K/W]")
    plt.legend()
    plt.xlim(graph_limits["R"]["xmin"],graph_limits["R"]["xmax"])
    plt.ylim(graph_limits["R"]["ymin"],graph_limits["R"]["ymax"])
    plt.tight_layout()
    
    path3=os.path.join(output_dir,"R_plot.png")
    plt.savefig(path3,dpi=300)
    plt.close()

    # === 最終行だけ温度分布グラフ ===
    last = len(df) - 1

    y1 = df.iloc[last, 1:5].astype(float).values
    y2 = df.iloc[last, 5:9].astype(float).values

    y1_sorted = np.sort(y1)[::-1]
    y2_sorted = np.sort(y2)[::-1]

    # 回帰直線
    a1, b1 = np.polyfit(x1, y1_sorted, 1)
    a2, b2 = np.polyfit(x2, y2_sorted, 1)

    # 外挿点
    x_ext1 = 4.5
    x_ext2 = 5 + d/10 - 0.5
    y_ext1 = a1 * x_ext1 + b1
    y_ext2 = a2 * x_ext2 + b2

    # R2
    y_pred1 = a1*x1 + b1
    y_pred2 = a2*x2 + b2

    SS1_res = np.sum((y1_sorted-y_pred1)**2)
    SS1_tot = np.sum((y1_sorted - np.mean(y1_sorted))**2)
    R2_1 = 1 - SS1_res/SS1_tot

    SS2_res = np.sum((y2_sorted-y_pred2)**2)
    SS2_tot = np.sum((y2_sorted - np.mean(y2_sorted))**2)
    R2_2 = 1 - SS2_res/SS2_tot

    plt.figure(figsize=(7,5))

    # 実測点
    plt.scatter(x1, y1_sorted, s=60, label="CH2–5 (sorted)")
    plt.scatter(x2, y2_sorted, s=60, label="CH7–10 (sorted)")

    # 外挿点
    # 上側 外挿点（青枠・白塗り）
    plt.scatter(
        x_ext1, y_ext1,
        s=60, marker="o",
        facecolors="white",
        edgecolors="tab:blue",
        linewidths=2,
        label="Upper surface"
    )

    # 下側 外挿点（オレンジ枠・白塗り）
    plt.scatter(
        x_ext2, y_ext2,
        s=60, marker="o",
        facecolors="white",
        edgecolors="orange",
        linewidths=2,
        label="Lower surface"
    )

    # 回帰線（外挿点まで）
    xx1 = np.linspace(min(x1), x_ext1, 100)
    xx2 = np.linspace(x_ext2, max(x2), 100)
    plt.plot(xx1, a1*xx1 + b1, linestyle="--", label="Fit Upper")
    plt.plot(xx2, a2*xx2 + b2, linestyle="--", label="Fit Lower")

    # 回帰式 + R2
    plt.text(
        0.05, 0.5,
        f"Upper: y = {a1:.3f} x + {b1:.3f}, R² = {R2_1:.4f}\n"
        f"Lower: y = {a2:.3f} x + {b2:.3f}, R² = {R2_2:.4f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.7)
    )

    plt.xlabel("Position")
    plt.ylabel("Temperature [°C]")
    plt.title("Temperature Distribution (Last Row)")
    plt.legend()
    plt.grid(True)
    plt.xlim(graph_limits["position"]["xlim"])
    plt.ylim(graph_limits["position"]["ylim"])
    plt.tight_layout()
    path4=os.path.join(output_dir,"温度分布.png")
    plt.savefig(path4,dpi=300)
    plt.close()
    
    # =============================
    # CSV保存
    # =============================

    result_path=os.path.join(output_dir,"result.csv")
    df_result.to_csv(result_path,index=False,encoding="shift_jis")

    return path1,path2,path3,path4,result_path,R_avg_1000,R_avg_2000,base_name


# =============================
# Streamlit UI
# =============================

st.title("熱抵抗解析 Webアプリ")

uploaded_file=st.file_uploader("CSVファイル")

r2=st.number_input("サンプル直径2r [cm] (サンプルが直方体の場合、2/√π=1.12837 cm)",value=0.8, step=0.000001)
d=st.number_input("サンプル厚みd [mm]",value=1.0, step=0.001)

st.subheader("グラフ設定")
time_min=st.number_input("Time_開始値 [s]",value=0)
time_max=st.number_input("Time_最大値 [s]",value=8000)
st.subheader("Rグラフ設定")

R_xmin=time_min
R_xmax=time_max

R_ymin=st.number_input("R_最小値 [mm²K/W]",value=0.0)
R_ymax=st.number_input("R_最大値 [mm²K/W]",value=1500)

st.subheader("上側銅ブロックのグラフ設定")
temp_xmin=time_min
temp_xmax=time_max

temp1_ymin=st.number_input("Temp_最小値 [℃]",value=20,key="temp1_ymin")
temp1_ymax=st.number_input("Temp_最大値 [℃]",value=110,key="temp1_ymax")

st.subheader("下側銅ブロックのグラフ設定")
temp2_ymin=st.number_input("Temp_最小値 [℃]",value=20,key="temp2_ymin")
temp2_ymax=st.number_input("Temp_最大値 [℃]",value=30,key="temp2_ymax")

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
        },
        
        "R":{
        "xmin":R_xmin,
        "xmax":R_xmax,
        "ymin":R_ymin,
        "ymax":R_ymax
        },
        
        "position": {
        "xlim":(0, 9),
        "ylim":(0, temp1_ymax)
        }

        }

        path1,path2,path3,path4,result_path,R_avg_1000,R_avg_2000,base_name=run_analysis(
        csv_path,tmp_dir,r2,d,graph_limits
        )
        st.session_state["analysis_done"]=True
        st.session_state["path1"]=path1
        st.session_state["path2"]=path2
        st.session_state["path3"]=path3
        st.session_state["path4"]=path4
        st.session_state["result_path"]=result_path
        st.session_state["base_name"]=base_name
        st.session_state["R1000"]=R_avg_1000
        st.session_state["R2000"]=R_avg_2000
        
if st.session_state.get("analysis_done",False):
    
    path1=st.session_state["path1"]
    path2=st.session_state["path2"]
    path3=st.session_state["path3"]
    path4=st.session_state["path4"]
    result_path=st.session_state["result_path"]
    base_name=st.session_state["base_name"]
    
    st.success("解析完了")
    st.metric("R_average (last 1000s)", f"{round(R_avg_1000,5)} mm²K/W")
    st.metric("R_average (last 2000s)", f"{round(R_avg_2000,5)} mm²K/W")
    
            
    st.image(path1)
    with open(path1,"rb") as f:
        st.download_button(
        "上側温度グラフPNGダウンロード",
         f,
        file_name="upper_temperature.png",
        mime="image/png"
        )
            
    st.image(path2)
    with open(path2,"rb") as f:
        st.download_button(
        "下側温度グラフPNGダウンロード",
        f,
        file_name="lower_temperature.png",
        mime="image/png"
        )
    st.image(path3)
    with open(path3,"rb") as f:
        st.download_button(
        "RグラフPNGダウンロード",
        f,
        file_name="R_plot.png",
        mime="image/png"
        )
        
    st.image(path4)
    with open(path4,"rb") as f:
        st.download_button(
        "温度分布グラフPNGダウンロード",
        f,
        file_name="温度分布_plot.png",
        mime="image/png"
        ) 
        
    with open(result_path, "rb") as f:
        st.download_button(
        "解析結果CSVダウンロード",
        f,
        file_name="result.csv",
        mime="text/csv"
        )
            
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as z:

        z.write(path1, arcname="upper_temperature.png")
        z.write(path2, arcname="lower_temperature.png")
        z.write(path3, arcname="R_plot.png")
        z.write(path4, arcname="温度分布_plot.png")
        z.write(result_path, arcname="回帰結果.csv")
    zip_buffer.seek(0)
        
    st.download_button(
        label="グラフPNG一括ダウンロード",
        data=zip_buffer,
        file_name=f"{base_name}_graphs.zip",
        mime="application/zip"
    )
            
if st.button("解析終了"):
    st.session_state.clear()
    
import streamlit as st
import math
import matplotlib.pyplot as plt
import numpy as np

st.markdown("""
                <div class="center-text">
                    <h2>✨ <span style="background: linear-gradient(90deg, #3f51b5, #2196f3);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        font-weight: bold;
                        margin-bottom: 20px;">Giải phương trình bậc 2</span>
                    </h2>
                </div>
            """, unsafe_allow_html=True)
st.markdown('<div class="center-text" style="margin-bottom: 15px">Tính nghiệm của phương trình bậc 2 dạng ( ax^2 + bx + c = 0) và vẽ đồ thị</div>', unsafe_allow_html=True)

def gptb2(a, b, c):
    if a == 0:
        if b == 0:
            if c == 0:
                return 'PTB1 có vô số nghiệm'
            else:
                return 'PTB1 vô nghiệm'
        else:
            x = -c/b
            return 'PTB1 có nghiệm x = %.2f' % x
    else:
        delta = b**2 - 4*a*c
        if delta < 0:
            return 'PTB2 vô nghiệm'
        elif delta == 0:
            x = -b / (2*a)
            return 'PTB2 có nghiệm kép x = %.2f' % x
        else:
            x1 = (-b + math.sqrt(delta))/(2*a)
            x2 = (-b - math.sqrt(delta))/(2*a)
            return 'PTB2 có nghiệm x1 = %.2f và x2 = %.2f' % (x1, x2)

def clear_input():
    st.session_state["nhap_a"] = 0.0
    st.session_state["nhap_b"] = 0.0
    st.session_state["nhap_c"] = 0.0

col1, col2 = st.columns(2)

with col1:
    with st.form(key='columns_in_form', clear_on_submit=False):
        a = st.number_input('Nhập a', key='nhap_a')
        b = st.number_input('Nhập b', key='nhap_b')
        c = st.number_input('Nhập c', key='nhap_c')
        c1, c2 = st.columns(2)
        with c1:
            btn_giai = st.form_submit_button('Giải')
        with c2:
            btn_xoa = st.form_submit_button('Xóa', on_click=clear_input)

        if btn_giai:
            ket_qua = gptb2(a, b, c)
            st.markdown('**Kết quả:** ' + ket_qua)
        else:
            st.markdown('**Kết quả:**')

with col2:
    x = np.linspace(-10, 10, 400)
    y = a * x**2 + b * x + c

    fig, ax = plt.subplots()
    ax.plot(x, y, label=f'y = {a}x² + {b}x + {c}', color='blue')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
# -*- coding: utf-8 -*-
"""
AIDEOM-VN — AI-Driven Decision Optimization Model for Vietnam
Web app giải 12 bài toán mô hình ra quyết định phát triển kinh tế Việt Nam
trong kỉ nguyên AI — dữ liệu thực 2020-2025.

Họ và tên : Đỗ Hoài Linh
Mã sinh viên: 23051285
Bài tập lớn: Các mô hình ra quyết định

Chạy:  streamlit run app.py
"""
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# CẤU HÌNH TRANG
# ============================================================
st.set_page_config(
    page_title="AIDEOM-VN — Mô hình ra quyết định",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Giao diện (dark, đỏ-hồng giống ảnh mẫu) — CHỮ SÁNG RÕ ----
st.markdown("""
<style>
:root { --vn-red:#ef4444; --vn-pink:#f43f5e; --txt:#e8eef7; }
.stApp { background:#0b1220; color:var(--txt); }
/* Ép chữ sáng cho mọi thành phần text của Streamlit */
.stApp, .stApp p, .stApp li, .stApp span, .stApp label,
.stMarkdown, .stMarkdown p, .stMarkdown li, .stCaption,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stText"], div[data-testid="stCaptionContainer"]{
  color:var(--txt) !important;
}
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5{
  color:#ffffff !important; letter-spacing:-.01em;
}
/* caption / chú thích nhạt hơn 1 chút nhưng vẫn đọc được */
.stCaption, div[data-testid="stCaptionContainer"] p{ color:#aebacb !important; }
/* công thức Latex */
.katex, .katex *{ color:#f1f5f9 !important; }
/* bảng dataframe */
[data-testid="stDataFrame"] *{ color:#e8eef7 !important; }
.big-red { color:var(--vn-pink); font-weight:800; }
.kpi-card{ background:#111a2e; border:1px solid #233047; border-radius:14px;
  padding:16px 18px; height:100%; }
.kpi-label{ color:#aebacb !important; font-size:.82rem; font-weight:600; }
.kpi-value{ color:var(--vn-pink) !important; font-size:1.8rem; font-weight:800; margin:.15rem 0; }
.kpi-delta{ display:inline-block; background:#0f2a1c; color:#5ef08a !important;
  border-radius:8px; padding:2px 8px; font-size:.75rem; font-weight:700; }
.lvl-chip{ display:inline-block; padding:3px 10px; border-radius:999px;
  font-size:.78rem; font-weight:700; margin-right:6px; }
.note{ background:#152138; border-left:3px solid var(--vn-pink);
  padding:10px 14px; border-radius:6px; color:#dbe4f0 !important; font-size:.9rem; }
.note b{ color:#ffffff !important; }
[data-testid="stSidebar"]{ background:#0e1626; }
[data-testid="stSidebar"] *{ color:#dbe4f0 !important; }
.sidebar-id{ background:#111a2e; border:1px solid #233047; border-radius:10px;
  padding:10px 12px; font-size:.82rem; color:#dbe4f0 !important; line-height:1.6; }
.sidebar-id b{ color:#ffffff !important; }
/* tab */
.stTabs [data-baseweb="tab-list"]{ gap:4px; }
.stTabs [data-baseweb="tab"]{ background:#111a2e; border-radius:8px 8px 0 0; padding:8px 14px; }
.stTabs [data-baseweb="tab"] p{ color:#cbd5e1 !important; font-weight:600; }
.stTabs [aria-selected="true"]{ background:#1c2742 !important; }
.stTabs [aria-selected="true"] p{ color:#ffffff !important; }
/* slider / radio label sáng */
.stSlider label, .stRadio label{ color:#e8eef7 !important; }
</style>
""", unsafe_allow_html=True)

PLOT_TMPL = "plotly_dark"
PALETTE = ["#22d3ee", "#fde68a", "#f87171", "#60a5fa", "#a78bfa", "#fb923c", "#4ade80"]


# ============================================================
# DỮ LIỆU — tạo CSV nếu thiếu, đọc lại bằng pandas (như notebook)
# ============================================================
def _write_csvs_if_missing():
    """Sinh 3 file CSV dữ liệu thực VN 2020-2025 nếu chưa tồn tại."""
    if not os.path.exists("vietnam_macro_2020_2025.csv"):
        pd.DataFrame({
            "year": [2020, 2021, 2022, 2023, 2024, 2025],
            "GDP_trillion_VND": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
            "population_million": [97.6, 98.5, 99.5, 100.3, 101.3, 102.3],
            "digital_economy_share_GDP_pct": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
            "labor_productivity_million_VND": [150.1, 171.8, 188.7, 199.3, 221.9, 245.0],
            "FDI_disbursed_billion_USD": [19.98, 19.74, 22.4, 23.18, 25.35, 27.6],
            "export_goods_billion_USD": [282.6, 336.3, 371.3, 354.7, 405.5, 475.0],
            "inflation_cpi_pct": [3.23, 1.84, 3.15, 3.25, 3.63, 3.5],
        }).to_csv("vietnam_macro_2020_2025.csv", index=False)

    if not os.path.exists("vietnam_sectors_2024.csv"):
        pd.DataFrame({
            "sector_name_vi": ["Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng",
                               "Khai khoáng", "Bán buôn-bán lẻ", "Tài chính-Ngân hàng",
                               "Logistics-Vận tải", "CNTT-Truyền thông", "Giáo dục-Đào tạo", "Y tế"],
            "sector_name_en": ["Agriculture", "Manufacturing", "Construction", "Mining",
                               "Wholesale-Retail", "Finance-Banking", "Logistics",
                               "ICT", "Education", "Healthcare"],
            "growth_rate_2024_pct": [3.27, 9.64, 7.45, -1.20, 7.10, 7.36, 9.93, 7.85, 6.42, 6.85],
            "gdp_share_2024_pct": [11.86, 24.10, 6.80, 2.50, 9.50, 5.10, 4.20, 3.80, 4.00, 3.20],
            "spillover_coef_0_1": [0.35, 0.78, 0.42, 0.30, 0.55, 0.85, 0.72, 0.92, 0.65, 0.60],
            "export_billion_USD": [40.5, 290.9, 2.5, 8.2, 5.5, 1.2, 3.1, 178.0, 0.0, 0.0],
            "labor_million": [13.20, 11.50, 4.80, 0.30, 7.80, 0.55, 1.95, 0.62, 2.15, 0.75],
            "ai_readiness_0_100": [15, 55, 20, 30, 48, 72, 42, 88, 38, 45],
            "automation_risk_pct": [18, 42, 25, 55, 38, 52, 35, 28, 22, 18],
            "rd_intensity_pct": [0.10, 0.55, 0.15, 0.20, 0.18, 0.85, 0.32, 1.20, 0.45, 0.30],
        }).to_csv("vietnam_sectors_2024.csv", index=False)

    if not os.path.exists("vietnam_regions_2024.csv"):
        pd.DataFrame({
            "region_name_vi": ["Trung du miền núi phía Bắc", "Đồng bằng sông Hồng",
                               "Bắc Trung Bộ + DH Trung Bộ", "Tây Nguyên",
                               "Đông Nam Bộ", "Đồng bằng sông Cửu Long"],
            "region_name_en": ["Northern Midlands and Mountains", "Red River Delta",
                               "North Central and South Central Coast", "Central Highlands",
                               "Southeast", "Mekong Delta"],
            "grdp_per_capita_million_VND": [57.0, 152.3, 87.5, 68.9, 158.9, 80.5],
            "fdi_registered_billion_USD": [3.5, 20.0, 8.2, 0.8, 18.5, 2.1],
            "digital_index_0_100": [38, 78, 55, 32, 82, 48],
            "ai_readiness_0_100": [22, 68, 40, 18, 75, 30],
            "trained_labor_pct": [21.5, 36.8, 27.5, 18.2, 42.5, 16.8],
            "rd_intensity_pct": [0.18, 0.85, 0.32, 0.15, 0.78, 0.22],
            "internet_penetration_pct": [72, 92, 84, 68, 94, 78],
            "gini_coef": [0.405, 0.358, 0.372, 0.412, 0.385, 0.392],
        }).to_csv("vietnam_regions_2024.csv", index=False)


@st.cache_data
def load_data():
    _write_csvs_if_missing()
    macro = pd.read_csv("vietnam_macro_2020_2025.csv").sort_values("year").reset_index(drop=True)
    sectors = pd.read_csv("vietnam_sectors_2024.csv")
    regions = pd.read_csv("vietnam_regions_2024.csv")
    return macro, sectors, regions


MACRO, SECTORS, REGIONS = load_data()

# Hằng số dùng lại (lấy từ đề bài / notebook)
K_HIST = np.array([16500, 17800, 19600, 21300, 23500, 25900])      # vốn tích lũy
L_HIST = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4])            # lao động (triệu)
D_HIST = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5])           # KTS/GDP (%)
AI_HIST = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])          # nghìn DN số
H_HIST = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2])          # LĐ qua ĐT (%)
COEF = dict(alpha=0.33, beta=0.42, gamma=0.10, delta=0.08, theta=0.07)


# ============================================================
# SIDEBAR — mục lục giống ảnh mẫu
# ============================================================
PAGES = [
    "🏠 Trang chủ",
    "🌱 Bài 1 — Cobb-Douglas + AI",
    "💰 Bài 2 — LP ngân sách số",
    "📊 Bài 3 — Priority 10 ngành",
    "🗺️ Bài 4 — LP ngành-vùng",
    "🎯 Bài 5 — MIP 15 dự án",
    "🏆 Bài 6 — TOPSIS 6 vùng",
    "🌐 Bài 7 — NSGA-II Pareto",
    "⏳ Bài 8 — Động 2026-2035",
    "👷 Bài 9 — Lao động & AI",
    "🎲 Bài 10 — Stochastic SP",
    "🤖 Bài 11 — Q-learning RL",
    "🇻🇳 Bài 12 — AIDEOM tích hợp",
]

with st.sidebar:
    st.markdown("### 🇻🇳 **AIDEOM-VN**")
    page = st.radio("Mục lục", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        '<div class="sidebar-id">'
        '<b>Họ và tên:</b> Đỗ Hoài Linh<br>'
        '<b>Mã sinh viên:</b> 23051285<br>'
        '<b>Bài tập lớn:</b> Các mô hình ra quyết định'
        '</div>', unsafe_allow_html=True)
    st.caption("Dữ liệu: NSO, MoST, MIC, MPI, WB, GII 2025")


# ============================================================
# HÀM TIỆN ÍCH
# ============================================================
def kpi(col, label, value, delta=None):
    html = f'<div class="kpi-card"><div class="kpi-label">{label}</div>' \
           f'<div class="kpi-value">{value}</div>'
    if delta:
        html += f'<span class="kpi-delta">↑ {delta}</span>'
    html += '</div>'
    col.markdown(html, unsafe_allow_html=True)


def section(title, sub=None):
    st.markdown(f"## {title}")
    if sub:
        st.caption(sub)


def pick_col(df, candidates, fallback_prefix="col"):
    """Trả về tên cột đầu tiên có trong df; nếu không có, tạo cột tạm."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ============================================================
# TRANG CHỦ
# ============================================================
def page_home():
    st.markdown("# 🇻🇳 AIDEOM-VN")
    st.markdown("### *AI-Driven Decision Optimization Model for Vietnam*")
    st.write("Web app giải **12 bài toán mô hình ra quyết định** phát triển kinh tế "
             "Việt Nam trong kỉ nguyên AI — dữ liệu thực 2020-2025.")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "GDP 2025", "514,0 tỷ USD", "8,02%")
    kpi(c2, "Kinh tế số / GDP", "≈19,5%", "1,2 đpt")
    kpi(c3, "FDI giải ngân 2025", "27,6 tỷ USD", "8,9%")
    kpi(c4, "GDP/người 2025", "5.026 USD", "6,9%")

    st.markdown("---")
    st.markdown("## 📚 12 bài toán theo 4 cấp độ")

    levels = [
        ("🟢 Cấp độ DỄ — Làm quen mô hình", "#16a34a", [
            ("Bài 1", "Hàm sản xuất Cobb-Douglas mở rộng + AI — Growth accounting, dự báo GDP 2030"),
            ("Bài 2", "LP phân bổ ngân sách 4 hạng mục — scipy.optimize, shadow price"),
            ("Bài 3", "Chỉ số ưu tiên 10 ngành — Min-max norm, weighted scoring, sensitivity"),
        ]),
        ("🟡 Cấp độ TRUNG BÌNH — Tối ưu cổ điển", "#ca8a04", [
            ("Bài 4", "LP phân bổ ngân sách số ngành-vùng (24 biến) — PuLP & CVXPY"),
            ("Bài 5", "MIP lựa chọn 15 dự án chuyển đổi số — biến nhị phân, precedence"),
            ("Bài 6", "TOPSIS xếp hạng 6 vùng theo ưu tiên AI — Entropy & AHP"),
        ]),
        ("🟠 Cấp độ KHÁ KHÓ — Đa mục tiêu & động", "#ea580c", [
            ("Bài 7", "NSGA-II tối ưu Pareto 4 mục tiêu — tăng trưởng/bao trùm/môi trường/an ninh"),
            ("Bài 8", "Tối ưu động liên thời gian 2026-2035 — quỹ đạo K, D, AI, H"),
            ("Bài 9", "Tác động AI tới lao động — NetJob ròng 8 ngành"),
        ]),
        ("🔴 Cấp độ KHÓ — Bất định & tích hợp", "#dc2626", [
            ("Bài 10", "Quy hoạch ngẫu nhiên 2 giai đoạn — VSS, EVPI, robust"),
            ("Bài 11", "Q-learning chính sách kinh tế thích nghi — MDP 81 trạng thái"),
            ("Bài 12", "Đồ án tích hợp AIDEOM-VN — 6 module / 4 tab dashboard"),
        ]),
    ]
    for title, color, items in levels:
        with st.expander(title, expanded=(color == "#16a34a")):
            for code, desc in items:
                st.markdown(
                    f'<span class="lvl-chip" style="background:{color}22;color:{color}">'
                    f'{code}</span> {desc}', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📂 Dữ liệu gốc Việt Nam 2020-2025")
    t1, t2, t3 = st.tabs(["Vĩ mô 2020-2025", "10 ngành 2024", "6 vùng KT-XH 2024"])
    with t1:
        st.dataframe(MACRO, use_container_width=True)
        fig = px.line(MACRO, x="year", y="GDP_trillion_VND", markers=True,
                      template=PLOT_TMPL, title="GDP Việt Nam (nghìn tỷ VND)")
        fig.update_traces(line_color="#f43f5e")
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        st.dataframe(SECTORS, use_container_width=True)
    with t3:
        st.dataframe(REGIONS, use_container_width=True)


# ============================================================
# BÀI 1 — COBB-DOUGLAS MỞ RỘNG
# ============================================================
def page_bai1():
    section("🌱 Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng với AI & số hóa",
            "Growth accounting trên dữ liệu Việt Nam 2020-2025 · dự báo GDP 2030")
    st.latex(r"Y_t = A_t \cdot K_t^{\alpha} L_t^{\beta} D_t^{\gamma} AI_t^{\delta} H_t^{\theta},"
             r"\quad \alpha+\beta+\gamma+\delta+\theta = 1")

    a, b, g, d, th = COEF["alpha"], COEF["beta"], COEF["gamma"], COEF["delta"], COEF["theta"]
    years = MACRO["year"].values
    Y = MACRO["GDP_trillion_VND"].values
    K, L, D, AI, H = K_HIST, L_HIST, D_HIST, AI_HIST, H_HIST

    # 1.4.1 — TFP
    A = Y / (K**a * L**b * D**g * AI**d * H**th)
    A_mean = A.mean()
    # 1.4.2 — dự báo + MAPE
    Y_hat = A_mean * (K**a * L**b * D**g * AI**d * H**th)
    mape = np.mean(np.abs((Y - Y_hat) / Y)) * 100

    c1, c2, c3 = st.columns(3)
    kpi(c1, "MAPE (Cobb-Douglas)", f"{mape:.2f}%")
    kpi(c2, "Ā (TFP trung bình)", f"{A_mean:.4f}")
    kpi(c3, "TFP tăng trưởng", f"{((A[-1]/A[0])**(1/5)-1)*100:.2f}%/năm")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["1.4.1 TFP A_t", "1.4.2 Dự báo & MAPE", "1.4.3 Phân rã tăng trưởng", "1.4.4 Dự báo 2030"])

    with tab1:
        st.write("Giải ngược (Solow residual): "
                 r"$A_t = Y_t / (K_t^{\alpha} L_t^{\beta} D_t^{\gamma} AI_t^{\delta} H_t^{\theta})$")
        dfA = pd.DataFrame({"Năm": years, "Y thực tế": Y, "A_t (TFP)": A.round(4)})
        st.dataframe(dfA, use_container_width=True)
        fig = px.line(dfA, x="Năm", y="A_t (TFP)", markers=True, template=PLOT_TMPL,
                      title="Năng suất nhân tố tổng hợp A_t 2020-2025")
        fig.update_traces(line_color="#22d3ee")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        dff = pd.DataFrame({"Năm": years, "Y thực tế": Y.round(1),
                            "Y dự báo": Y_hat.round(1),
                            "Sai số %": ((Y_hat - Y) / Y * 100).round(2)})
        st.dataframe(dff, use_container_width=True)
        fig = go.Figure()
        fig.add_bar(x=years, y=Y, name="Y thực tế", marker_color="#60a5fa")
        fig.add_trace(go.Scatter(x=years, y=Y_hat, name="Y dự báo",
                                 mode="lines+markers", line_color="#f43f5e"))
        fig.update_layout(template=PLOT_TMPL, title=f"Thực tế vs dự báo (MAPE = {mape:.2f}%)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        n = 5
        gY = (np.log(Y[-1]) - np.log(Y[0])) / n
        gs = {"TFP (A)": (np.log(A[-1]) - np.log(A[0])) / n,
              "Vốn (K)": a * (np.log(K[-1]) - np.log(K[0])) / n,
              "Lao động (L)": b * (np.log(L[-1]) - np.log(L[0])) / n,
              "Số hóa (D)": g * (np.log(D[-1]) - np.log(D[0])) / n,
              "AI": d * (np.log(AI[-1]) - np.log(AI[0])) / n,
              "Nhân lực số (H)": th * (np.log(H[-1]) - np.log(H[0])) / n}
        dfd = pd.DataFrame({
            "Yếu tố": list(gs.keys()),
            "Đóng góp (%)": [v * 100 for v in gs.values()],
            "Tỷ lệ (%)": [v / gY * 100 for v in gs.values()]})
        st.dataframe(dfd.round(3), use_container_width=True)
        fig = px.bar(dfd, x="Yếu tố", y="Đóng góp (%)", template=PLOT_TMPL,
                     color="Yếu tố", color_discrete_sequence=PALETTE,
                     title=f"Phân rã đóng góp tăng trưởng GDP bình quân năm = {gY*100:.2f}%")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.write("Kịch bản 2030: D=30%, AI=100 nghìn DN, H=35%, K +6%/năm, TFP +1,2%/năm.")
        cc1, cc2 = st.columns(2)
        l_growth = cc1.slider("Tăng trưởng L (%/năm)", 0.0, 6.0, 0.5, 0.5)
        k_growth = cc2.slider("Tăng trưởng K (%/năm)", 3.0, 10.0, 6.0, 0.5)
        K30 = K[-1] * (1 + k_growth / 100)**5
        L30 = L[-1] * (1 + l_growth / 100)**5
        A30 = A[-1] * 1.012**5
        Y30 = A30 * (K30**a * L30**b * 30.0**g * 100.0**d * 35.0**th)
        cc = st.columns(3)
        kpi(cc[0], "GDP 2030 dự báo", f"{Y30:,.0f} ng.tỷ")
        kpi(cc[1], "Tăng trưởng BQ 25-30", f"{((Y30/Y[-1])**(1/5)-1)*100:.2f}%/năm")
        kpi(cc[2], "GDP/người 2030 (~110tr)",
            f"{Y30*1e12/(110e6)/25500:,.0f} USD")
        proj = [Y[-1]]
        for t in range(5):
            proj.append(Y[-1] * (Y30 / Y[-1])**((t + 1) / 5))
        fig = px.line(x=list(range(2025, 2031)), y=proj, markers=True,
                      template=PLOT_TMPL, title="Quỹ đạo GDP 2025→2030",
                      labels={"x": "Năm", "y": "GDP (ng.tỷ VND)"})
        fig.update_traces(line_color="#f43f5e")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="note"><b>Thảo luận:</b> TFP tăng đều giai đoạn 2020-2025 → '
                'tăng trưởng có cải thiện về chất. Trong các yếu tố mới, <b>số hóa (D)</b> và '
                '<b>AI</b> đóng góp nhanh nhất nhờ tăng trưởng cao từ nền thấp. Mục tiêu 30% '
                'KTS/GDP 2030 khả thi nếu duy trì đầu tư D & H đồng thời.</div>',
                unsafe_allow_html=True)


# ============================================================
# BÀI 2 — LP NGÂN SÁCH 4 HẠNG MỤC
# ============================================================
def page_bai2():
    from scipy.optimize import linprog
    section("💰 Bài 2 — Phân bổ ngân sách 4 hạng mục đầu tư số",
            "LP đơn giản · scipy.optimize.linprog & PuLP · shadow price")
    st.latex(r"\max Z = 0.85x_1 + 1.20x_2 + 0.95x_3 + 1.35x_4")
    st.caption("x₁ hạ tầng số · x₂ AI & dữ liệu · x₃ nhân lực số · x₄ R&D công nghệ (nghìn tỷ VND)")

    def solve(B=100, x3min=20):
        c = [-0.85, -1.20, -0.95, -1.35]
        A_ub = [[1, 1, 1, 1], [-1, 0, 0, 0], [0, -1, 0, 0],
                [0, 0, -1, 0], [0, 0, 0, -1], [0.35, -0.65, 0.35, -0.65]]
        b_ub = [B, -25, -15, -x3min, -10, 0]
        return linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * 4, method="highs")

    tab1, tab2, tab3 = st.tabs(["2.4.1-2 Lời giải & shadow price",
                                "2.4.3 Độ nhạy ngân sách", "2.4.4 Ưu tiên nhân lực"])
    with tab1:
        res = solve()
        names = ["x₁ Hạ tầng số", "x₂ AI & dữ liệu", "x₃ Nhân lực số", "x₄ R&D công nghệ"]
        df = pd.DataFrame({"Hạng mục": names, "Phân bổ (ng.tỷ)": res.x.round(2)})
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.dataframe(df, use_container_width=True)
        with c2:
            kpi(st, "Z* (GDP tăng thêm)", f"{-res.fun:.2f} ng.tỷ")
            st.write(f"Tỷ trọng AI+R&D = **{(res.x[1]+res.x[3])/res.x.sum()*100:.1f}%**")
        fig = px.pie(df, names="Hạng mục", values="Phân bổ (ng.tỷ)", hole=.45,
                     template=PLOT_TMPL, color_discrete_sequence=PALETTE)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="note"><b>Shadow price ngân sách tổng = 1,35:</b> mỗi nghìn tỷ '
                    'tăng thêm tạo 1,35 nghìn tỷ GDP (= hệ số biên của R&D). Đây là cận trên hợp '
                    'lý của chi phí cơ hội vốn công.</div>', unsafe_allow_html=True)
    with tab2:
        Bs = np.arange(100, 201, 10)
        Zs = [-solve(B).fun for B in Bs]
        fig = px.line(x=Bs, y=Zs, markers=True, template=PLOT_TMPL,
                      labels={"x": "Ngân sách tổng (ng.tỷ)", "y": "Z* (ng.tỷ GDP)"},
                      title="Đường cong Z*(B)")
        fig.update_traces(line_color="#f43f5e")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame({"B": [100, 120, 140],
                                   "Z*": [round(-solve(b).fun, 2) for b in [100, 120, 140]]}),
                     use_container_width=True)
    with tab3:
        x3min = st.slider("Ràng buộc nhân lực số tối thiểu x₃ ≥", 20, 40, 30, 1)
        res = solve(x3min=x3min)
        if res.success:
            st.success(f"Khả thi · Z* = {-res.fun:.2f} ng.tỷ "
                       f"(giảm {(-solve().fun)-(-res.fun):.2f} so với x₃≥20)")
            st.dataframe(pd.DataFrame(
                {"Hạng mục": ["x₁", "x₂", "x₃", "x₄"], "Phân bổ": res.x.round(2)}),
                use_container_width=True)
        else:
            st.error("Bài toán KHÔNG khả thi với ràng buộc này.")


# ============================================================
# BÀI 3 — PRIORITY 10 NGÀNH
# ============================================================
def page_bai3():
    section("📊 Bài 3 — Chỉ số ưu tiên ngành cho 10 ngành Việt Nam",
            "Chuẩn hóa min-max · weighted scoring · phân tích độ nhạy")
    df = SECTORS.copy()
    # Dò tên cột (tương thích cả file dữ liệu thật lẫn file tự sinh)
    c_name = pick_col(df, ["sector_name_vi", "sector_name_en", "sector_name", "sector", "name"])
    c_growth = pick_col(df, ["growth_rate_2024_pct", "growth_rate", "growth"])
    c_share = pick_col(df, ["gdp_share_2024_pct", "gdp_share", "share"])
    c_spill = pick_col(df, ["spillover_coef_0_1", "spillover", "spillover_coef"])
    c_exp = pick_col(df, ["export_billion_USD", "export", "export_billion"])
    c_labor = pick_col(df, ["labor_million", "labor", "employment_million"])
    c_ai = pick_col(df, ["ai_readiness_0_100", "ai_readiness", "ai"])
    c_risk = pick_col(df, ["automation_risk_pct", "automation_risk", "risk"])

    missing = [n for n, c in [("tên ngành", c_name), ("tăng trưởng", c_growth),
               ("tỷ trọng GDP", c_share), ("lan tỏa", c_spill), ("xuất khẩu", c_exp),
               ("lao động", c_labor), ("AI readiness", c_ai), ("rủi ro", c_risk)] if c is None]
    if missing:
        st.error("File `vietnam_sectors_2024.csv` thiếu cột: " + ", ".join(missing) +
                 f". Các cột hiện có: {list(df.columns)}")
        return
    if c_name is None or df[c_name].dtype.kind in "if":
        df["_name"] = [f"Ngành {i+1}" for i in range(len(df))]
        c_name = "_name"

    GDP24 = 11511.9
    df["labor_productivity"] = (df[c_share] / 100) * GDP24 / df[c_labor]
    cols_good = [c_growth, "labor_productivity", c_spill, c_exp, c_labor, c_ai]

    def norm_good(x): return (x - x.min()) / (x.max() - x.min())
    def norm_bad(x): return (x.max() - x) / (x.max() - x.min())
    Xg = df[cols_good].apply(norm_good)
    Xb = norm_bad(df[c_risk])

    tab0, tab1, tab2, tab3 = st.tabs(["3.4.1 Ma trận chuẩn hóa", "3.4.2 Xếp hạng mặc định",
                                      "3.4.3 Độ nhạy w_AI", "3.4.4 Hai bộ trọng số"])
    with tab0:
        st.write("Chuẩn hóa min-max về [0,1]; riêng **Risk** đảo dấu (rủi ro thấp = điểm cao).")
        norm = Xg.copy()
        norm.columns = ["Tăng trưởng", "Năng suất", "Lan tỏa", "Xuất khẩu", "Việc làm", "AI Ready"]
        norm.insert(0, "Ngành", df[c_name].values)
        norm["Risk (đảo)"] = Xb.values
        st.dataframe(norm.round(4), use_container_width=True, hide_index=True)
    with tab1:
        w_raw = np.array([0.15, 0.15, 0.20, 0.15, 0.10, 0.20]); wr = 0.15
        tot = w_raw.sum() + wr
        pr = Xg.values @ (w_raw / tot) + (wr / tot) * Xb.values
        rk = pd.DataFrame({"Ngành": df[c_name], "Priority": pr.round(4)}) \
            .sort_values("Priority", ascending=False).reset_index(drop=True)
        rk.index += 1
        st.dataframe(rk, use_container_width=True)
        fig = px.bar(rk, x="Priority", y="Ngành", orientation="h", template=PLOT_TMPL,
                     color="Priority", color_continuous_scale="Reds",
                     title="Xếp hạng ưu tiên 10 ngành (trọng số mặc định)")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        w_base = np.array([0.15, 0.15, 0.20, 0.15, 0.10]); wr = 0.15
        rng = np.arange(0.05, 0.45, 0.05)
        heat = []
        for wai in rng:
            rem = 1 - wai - wr
            wsc = w_base * (rem / w_base.sum())
            heat.append(Xg.values @ np.append(wsc, wai) + wr * Xb.values)
        heat = np.array(heat)
        fig = px.imshow(heat, x=[f"N{i+1}" for i in range(10)],
                        y=[f"{w:.2f}" for w in rng], aspect="auto",
                        color_continuous_scale="YlOrRd", template=PLOT_TMPL,
                        labels=dict(x="Ngành", y="w_AI", color="Priority"),
                        title="Heatmap Priority theo w_AI")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("N1..N10 = " + ", ".join(f"N{i+1}:{n}" for i, n in
                   enumerate(df[c_name])))
    with tab3:
        wg = np.array([0.25, 0.25, 0.10, 0.25, 0.05, 0.05]); wg_r = 0.05
        wi = np.array([0.05, 0.10, 0.25, 0.05, 0.25, 0.10]); wi_r = 0.20
        pg = Xg.values @ wg + wg_r * Xb.values
        pi = Xg.values @ wi + wi_r * Xb.values
        comp = pd.DataFrame({"Ngành": df[c_name],
                             "P. Tăng trưởng": pg.round(4), "P. Bao trùm": pi.round(4)})
        c1, c2 = st.columns(2)
        c1.write("**Top-3 Tăng trưởng**")
        c1.dataframe(comp.nlargest(3, "P. Tăng trưởng")[["Ngành", "P. Tăng trưởng"]],
                     hide_index=True, use_container_width=True)
        c2.write("**Top-3 Bao trùm**")
        c2.dataframe(comp.nlargest(3, "P. Bao trùm")[["Ngành", "P. Bao trùm"]],
                     hide_index=True, use_container_width=True)
        fig = go.Figure()
        fig.add_bar(y=comp["Ngành"], x=comp["P. Tăng trưởng"], name="Tăng trưởng",
                    orientation="h", marker_color="#60a5fa")
        fig.add_bar(y=comp["Ngành"], x=comp["P. Bao trùm"], name="Bao trùm",
                    orientation="h", marker_color="#f87171")
        fig.update_layout(template=PLOT_TMPL, barmode="group",
                          title="So sánh 2 định hướng trọng số")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# BÀI 4 — LP NGÀNH-VÙNG
# ============================================================
REGIONS_VI = ['Trung du miền núi', 'ĐB sông Hồng', 'Bắc Trung Bộ',
              'Tây Nguyên', 'Đông Nam Bộ', 'ĐB sông Cửu Long']
REG = ['NMM', 'RRD', 'NCC', 'CH', 'SE', 'MD']
ITEMS = ['I', 'D', 'AI', 'H']
BETA_RJ = {
    ('NMM', 'I'): 1.15, ('NMM', 'D'): 0.85, ('NMM', 'AI'): 0.55, ('NMM', 'H'): 1.30,
    ('RRD', 'I'): 0.95, ('RRD', 'D'): 1.25, ('RRD', 'AI'): 1.40, ('RRD', 'H'): 1.05,
    ('NCC', 'I'): 1.05, ('NCC', 'D'): 0.95, ('NCC', 'AI'): 0.85, ('NCC', 'H'): 1.15,
    ('CH', 'I'): 1.20, ('CH', 'D'): 0.75, ('CH', 'AI'): 0.45, ('CH', 'H'): 1.35,
    ('SE', 'I'): 0.90, ('SE', 'D'): 1.30, ('SE', 'AI'): 1.55, ('SE', 'H'): 1.00,
    ('MD', 'I'): 1.10, ('MD', 'D'): 0.85, ('MD', 'AI'): 0.65, ('MD', 'H'): 1.25}
def _reg_col(cands):
    for c in cands:
        if c in REGIONS.columns:
            return c
    return None

REG_NAME_COL = _reg_col(["region_name_vi", "region_name_en", "region_name", "region", "name"])
REG_DIGITAL_COL = _reg_col(["digital_index_0_100", "digital_index", "digital"])
_D0_default = [38, 78, 55, 32, 82, 48]
if REG_DIGITAL_COL and len(REGIONS) == 6:
    D0_REG = dict(zip(REG, REGIONS[REG_DIGITAL_COL].values))
else:
    D0_REG = dict(zip(REG, _D0_default))


def _solve_lp4(with_equity=True):
    import pulp
    gamma_val, lam = 0.002, 0.6
    m = pulp.LpProblem('LP4', pulp.LpMaximize)
    x = pulp.LpVariable.dicts('x', (REG, ITEMS), lowBound=0)
    m += pulp.lpSum(BETA_RJ[(r, j)] * x[r][j] for r in REG for j in ITEMS)
    m += pulp.lpSum(x[r][j] for r in REG for j in ITEMS) <= 50000
    for r in REG:
        m += pulp.lpSum(x[r][j] for j in ITEMS) >= 5000
        m += pulp.lpSum(x[r][j] for j in ITEMS) <= 12000
    m += pulp.lpSum(x[r]['H'] for r in REG) >= 12000
    if with_equity:
        M = pulp.LpVariable('Dmax')
        for r in REG:
            m += D0_REG[r] + gamma_val * x[r]['D'] <= M
            m += D0_REG[r] + gamma_val * x[r]['D'] >= lam * M
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    mat = np.array([[x[r][j].value() for j in ITEMS] for r in REG])
    return mat, pulp.value(m.objective)


def page_bai4():
    section("🗺️ Bài 4 — LP phân bổ ngân sách số ngành-vùng",
            "24 biến · ràng buộc công bằng vùng miền · PuLP (CBC)")
    eq, noeq = _solve_lp4(True), _solve_lp4(False)
    x_opt, Z = eq
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Z* (có công bằng)", f"{Z:,.0f} tỷ")
    kpi(c2, "Z* (không công bằng)", f"{noeq[1]:,.0f} tỷ")
    kpi(c3, "Chi phí công bằng", f"{noeq[1]-Z:,.0f} tỷ")

    tab1, tabc, tab2 = st.tabs(["4.4.1-3 Phân bổ tối ưu (PuLP)",
                                "4.4.2 Đối chiếu CVXPY", "4.4.4 Chi phí công bằng vùng"])
    with tab1:
        dfm = pd.DataFrame(x_opt, index=REGIONS_VI, columns=ITEMS).round(0)
        dfm["Tổng"] = dfm.sum(axis=1)
        st.dataframe(dfm, use_container_width=True)
        fig = px.imshow(x_opt, x=ITEMS, y=REGIONS_VI, aspect="auto",
                        color_continuous_scale="YlOrRd", template=PLOT_TMPL,
                        text_auto=".0f", title=f"Heatmap phân bổ tối ưu (Z*={Z:,.0f} tỷ)")
        st.plotly_chart(fig, use_container_width=True)
    with tabc:
        try:
            import cvxpy as cp
            beta_mat = np.array([[BETA_RJ[(r, j)] for j in ITEMS] for r in REG])
            xv = cp.Variable((6, 4), nonneg=True)
            rs = cp.sum(xv, axis=1)
            cons = [cp.sum(xv) <= 50000, rs >= 5000, rs <= 12000, cp.sum(xv[:, 3]) >= 12000]
            D0v = np.array([D0_REG[r] for r in REG])
            Dn = D0v + 0.002 * xv[:, 1]
            Mc = cp.Variable()
            cons += [Dn <= Mc, Dn >= 0.6 * Mc]
            prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(beta_mat, xv))), cons)
            for sv in ["CLARABEL", "ECOS", "SCS"]:
                try:
                    prob.solve(solver=getattr(cp, sv)); 
                    if prob.status.startswith("optimal"): break
                except Exception: continue
            c1, c2 = st.columns(2)
            kpi(c1, "Z* PuLP (CBC)", f"{Z:,.0f}")
            kpi(c2, "Z* CVXPY", f"{prob.value:,.0f}")
            st.success(f"Chênh lệch PuLP vs CVXPY = {abs(Z - prob.value):.2f} tỷ → "
                       f"hai solver cho kết quả trùng khớp.")
            st.dataframe(pd.DataFrame(xv.value, index=REGIONS_VI, columns=ITEMS).round(0),
                         use_container_width=True)
        except ImportError:
            st.warning("Chưa cài CVXPY (`pip install cvxpy`). PuLP Z* = "
                       f"{Z:,.0f} tỷ — kết quả tham chiếu.")
    with tab2:
        fig = go.Figure()
        fig.add_bar(x=comp["Vùng"], y=comp["Có công bằng"], name="Có công bằng",
                    marker_color="#60a5fa")
        fig.add_bar(x=comp["Vùng"], y=comp["Không công bằng"], name="Không công bằng",
                    marker_color="#f87171")
        fig.update_layout(template=PLOT_TMPL, barmode="group",
                          title="Tổng ngân sách theo vùng")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="note"><b>Thảo luận:</b> Bỏ ràng buộc công bằng, vốn chảy mạnh '
                    'về ĐNB & ĐBSH (β_AI cao nhất). Chi phí công bằng = phần GDP gain hi sinh '
                    'để nâng chỉ số số hóa vùng yếu lên ≥60% vùng tốt nhất.</div>',
                    unsafe_allow_html=True)


# ============================================================
# BÀI 5 — MIP 15 DỰ ÁN
# ============================================================
def page_bai5():
    from pulp import (LpProblem, LpMaximize, LpVariable, lpSum, value,
                      PULP_CBC_CMD, LpStatus)
    section("🎯 Bài 5 — MIP lựa chọn dự án chuyển đổi số",
            "15 dự án · biến nhị phân · loại trừ, tiên quyết, ngân sách đa năm")
    P = list(range(1, 16))
    C = {1: 12000, 2: 11500, 3: 18000, 4: 4500, 5: 3200, 6: 5800, 7: 6500, 8: 15000,
         9: 2500, 10: 7200, 11: 4800, 12: 8500, 13: 20000, 14: 3800, 15: 1500}
    C1 = {1: 8500, 2: 7500, 3: 12000, 4: 3500, 5: 2500, 6: 4000, 7: 4500, 8: 9000,
          9: 1800, 10: 5000, 11: 3500, 12: 5500, 13: 13000, 14: 2800, 15: 1200}
    B = {1: 21500, 2: 20800, 3: 32500, 4: 9200, 5: 6800, 6: 11400, 7: 12200, 8: 28500,
         9: 5800, 10: 13800, 11: 8500, 12: 16200, 13: 35000, 14: 7500, 15: 3800}
    names = {1: 'TT dữ liệu Hòa Lạc', 2: 'TT dữ liệu phía Nam', 3: '5G toàn quốc',
             4: 'VNeID 2.0', 5: 'Cổng DVC v3', 6: 'Y tế số', 7: 'Giáo dục số K-12',
             8: 'TT AI + supercomputing', 9: 'Fintech sandbox', 10: 'Logistics thông minh',
             11: 'Nông nghiệp số ĐBSCL', 12: 'Đào tạo 50K kỹ sư AI',
             13: 'Khu CN bán dẫn BN-BG', 14: 'An ninh mạng SOC', 15: 'Open Data'}
    fields = {1: 'ht', 2: 'ht', 3: 'ht', 4: 'cp', 5: 'cp', 6: 'yt', 7: 'gd', 8: 'ai',
              9: 'tc', 10: 'lg', 11: 'nn', 12: 'nl', 13: 'bd', 14: 'an', 15: 'dl'}
    prob = {'ht': .85, 'cp': .75, 'ai': .65, 'bd': .65, 'yt': .8, 'gd': .8, 'tc': .8,
            'lg': .8, 'nn': .8, 'nl': .8, 'an': .8, 'dl': .8}

    def solve(BT=80000, B12=40000, use_exp=False, force12=False):
        m = LpProblem('sel', LpMaximize)
        y = LpVariable.dicts('y', P, cat='Binary')
        m += lpSum((prob[fields[i]] if use_exp else 1) * B[i] * y[i] for i in P)
        m += lpSum(C[i] * y[i] for i in P) <= BT
        m += lpSum(C1[i] * y[i] for i in P) <= B12
        if force12:
            m += y[1] >= 1; m += y[2] >= 1
        else:
            m += y[1] + y[2] <= 1
        m += y[8] <= y[12]; m += y[13] <= y[12]
        m += y[4] + y[5] >= 1; m += y[14] >= 1
        m += lpSum(y[i] for i in P) >= 7; m += lpSum(y[i] for i in P) <= 11
        m.solve(PULP_CBC_CMD(msg=False))
        sel = [i for i in P if y[i].value() > 0.5]
        return sel, sum(C[i] for i in sel), value(m.objective), LpStatus[m.status]

    tab1, tab2, tabf, tab3 = st.tabs(["5.4.1 Lời giải cơ sở", "5.4.2 Nới ngân sách",
                                      "5.4.3 Bắt buộc P1+P2", "5.4.4 Lợi ích kỳ vọng"])
    with tab1:
        sel, tc, Z, _ = solve()
        c1, c2, c3 = st.columns(3)
        kpi(c1, "Số dự án chọn", f"{len(sel)}")
        kpi(c2, "Tổng chi phí", f"{tc:,} tỷ")
        kpi(c3, "Tổng lợi ích Z*", f"{Z:,.0f} tỷ")
        df = pd.DataFrame([{"Mã": f"P{i}", "Tên": names[i], "Chi phí": C[i],
                            "NPV": B[i], "B/C": round(B[i]/C[i], 2)} for i in sel])
        st.dataframe(df, use_container_width=True, hide_index=True)
    with tab2:
        BT = st.slider("Ngân sách tổng (tỷ VND)", 80000, 120000, 100000, 5000)
        sel, tc, Z, _ = solve(BT=BT)
        sel0, _, Z0, _ = solve()
        st.success(f"Chọn {len(sel)} dự án · Z* = {Z:,.0f} tỷ (Δ{Z-Z0:+,.0f} so với 80.000)")
        st.write("Dự án thêm mới: " + ", ".join(f"P{i}" for i in set(sel) - set(sel0)) or "—")
        st.write("Đã chọn: " + ", ".join(f"P{i}" for i in sel))
    with tab3:
        sel0, _, Z0, _ = solve()
        sel4, tc4, Z4, _ = solve(use_exp=True)
        c1, c2 = st.columns(2)
        kpi(c1, "Deterministic Z*", f"{Z0:,.0f} tỷ")
        kpi(c2, "Expected E[Z]", f"{Z4:,.0f} tỷ")
        st.write("Bị loại khi tính rủi ro: " +
                 (", ".join(f"P{i}" for i in set(sel0) - set(sel4)) or "—"))
        st.write("Được thêm khi tính rủi ro: " +
                 (", ".join(f"P{i}" for i in set(sel4) - set(sel0)) or "—"))
    with tabf:
        sel0, _, Z0, _ = solve()
        sel3, tc3, Z3, stt = solve(force12=True)
        if stt == "Optimal":
            st.success(f"Vẫn khả thi · Z* = {Z3:,.0f} tỷ "
                       f"(giảm {Z0-Z3:,.0f} so với cơ sở do buộc cả P1 và P2)")
            st.write("Đã chọn: " + ", ".join(f"P{i}" for i in sel3))
            kpi(st, "Tổng chi phí", f"{tc3:,} tỷ")
        else:
            st.error("Bài toán KHÔNG khả thi khi bắt buộc cả P1 và P2.")


# ============================================================
# BÀI 6 — TOPSIS 6 VÙNG
# ============================================================
TOPSIS_CRIT_CANDS = [
    (["grdp_per_capita_million_VND", "grdp_per_capita", "grdp"], True),
    (["fdi_registered_billion_USD", "fdi_registered", "fdi"], True),
    (["digital_index_0_100", "digital_index", "digital"], True),
    (["ai_readiness_0_100", "ai_readiness", "ai"], True),
    (["trained_labor_pct", "trained_labor", "labor_trained"], True),
    (["rd_intensity_pct", "rd_intensity", "rd"], True),
    (["internet_penetration_pct", "internet_penetration", "internet"], True),
    (["gini_coef", "gini"], False),
]
TOPSIS_LBL_ALL = ['GRDP/N', 'FDI', 'Digital', 'AI', 'LĐĐT', 'R&D', 'Internet', 'Gini']
_resolved = [(_reg_col(c), lbl, ben) for (c, ben), lbl in zip(TOPSIS_CRIT_CANDS, TOPSIS_LBL_ALL)]
TOPSIS_CRIT = [c for c, _, _ in _resolved if c is not None]
TOPSIS_LBL = [lbl for c, lbl, _ in _resolved if c is not None]
IS_BENEFIT = np.array([ben for c, _, ben in _resolved if c is not None])


def _topsis(X, w, isb):
    R = X / np.sqrt((X**2).sum(0))
    V = R * w
    A_s = np.where(isb, V.max(0), V.min(0))
    A_n = np.where(isb, V.min(0), V.max(0))
    S_s = np.sqrt(((V - A_s)**2).sum(1))
    S_n = np.sqrt(((V - A_n)**2).sum(1))
    return S_n / (S_s + S_n)


def _entropy_w(X):
    P = X / X.sum(0)
    k = 1.0 / np.log(len(X))
    E = -k * np.nansum(P * np.log(P + 1e-12), 0)
    d = 1 - E
    return d / d.sum()


def page_bai6():
    section("🏆 Bài 6 — TOPSIS xếp hạng 6 vùng theo ưu tiên đầu tư AI",
            "Chuẩn hóa vector · trọng số chuyên gia vs Entropy")
    X = REGIONS[TOPSIS_CRIT].values.astype(float)
    rn = REGIONS[REG_NAME_COL].values if REG_NAME_COL else np.array(REGIONS_VI)
    w_exp = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])
    C_exp = _topsis(X, w_exp, IS_BENEFIT)
    w_ent = _entropy_w(X)
    C_ent = _topsis(X, w_ent, IS_BENEFIT)

    tab1, tab2, tab3, tab4 = st.tabs(["6.4.1 Trọng số chuyên gia",
                                      "6.4.2 Entropy vs Expert", "6.4.3 Độ nhạy w_AI",
                                      "6.4.4 AHP vs TOPSIS"])
    with tab1:
        df = pd.DataFrame({"Vùng": rn, "C*": C_exp.round(4)}) \
            .sort_values("C*", ascending=False).reset_index(drop=True)
        df.index += 1
        st.dataframe(df, use_container_width=True)
        fig = px.bar(df, x="C*", y="Vùng", orientation="h", template=PLOT_TMPL,
                     color="C*", color_continuous_scale="Reds",
                     title="Xếp hạng TOPSIS (trọng số chuyên gia)")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        comp = pd.DataFrame({"Vùng": rn, "C* Expert": C_exp.round(4),
                             "C* Entropy": C_ent.round(4)})
        st.dataframe(comp, use_container_width=True, hide_index=True)
        fig = go.Figure()
        fig.add_bar(y=rn, x=C_exp, name="Expert", orientation="h", marker_color="#60a5fa")
        fig.add_bar(y=rn, x=C_ent, name="Entropy", orientation="h", marker_color="#fb923c")
        fig.update_layout(template=PLOT_TMPL, barmode="group", title="Expert vs Entropy")
        st.plotly_chart(fig, use_container_width=True)
        we = pd.DataFrame({"Tiêu chí": TOPSIS_LBL, "Entropy w": w_ent.round(4)})
        st.caption("Trọng số Entropy (khách quan):")
        st.dataframe(we, use_container_width=True, hide_index=True)
    with tab3:
        rng = np.arange(0.10, 0.45, 0.05)
        heat = []
        for wai in rng:
            wg = 0.10
            rem = 1 - wai - wg
            wb = np.array([0.10, 0.10, 0.15, 0.15, 0.15, 0.05])
            wsc = wb * (rem / wb.sum())
            wfull = np.insert(wsc, 3, wai)
            wfull = np.append(wfull, wg)
            heat.append(_topsis(X, wfull, IS_BENEFIT))
        heat = np.array(heat)
        fig = px.imshow(heat, x=[f"V{i+1}" for i in range(6)],
                        y=[f"{w:.2f}" for w in rng], aspect="auto", text_auto=".3f",
                        color_continuous_scale="YlOrRd", template=PLOT_TMPL,
                        labels=dict(x="Vùng", y="w_AI", color="C*"),
                        title="C* theo w_AI (0.10→0.40)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("V1..V6 = " + ", ".join(f"V{i+1}:{n}" for i, n in enumerate(rn)))
    with tab4:
        ahp = np.array([
            [1, 1, 1/3, 1/5, 1/3, 1/3, 3, 3], [1, 1, 1/3, 1/5, 1/3, 1/3, 3, 3],
            [3, 3, 1, 1/2, 1, 1, 5, 5], [5, 5, 2, 1, 2, 2, 7, 7],
            [3, 3, 1, 1/2, 1, 1, 5, 5], [3, 3, 1, 1/2, 1, 1, 5, 5],
            [1/3, 1/3, 1/5, 1/7, 1/5, 1/5, 1, 1], [1/3, 1/3, 1/5, 1/7, 1/5, 1/5, 1, 1]])
        n = 8
        gm = np.prod(ahp, axis=1) ** (1 / n)
        w_ahp = gm / gm.sum()
        lam = np.mean((ahp @ w_ahp) / w_ahp)
        CI = (lam - n) / (n - 1); CR = CI / 1.41
        C_ahp = _topsis(X, w_ahp, IS_BENEFIT)
        st.write(f"**Kiểm tra nhất quán:** λ_max={lam:.3f}, CI={CI:.3f}, "
                 f"**CR={CR:.3f}** ({'✅ nhất quán' if CR < 0.10 else '⚠️ chưa nhất quán'})")
        cmp = pd.DataFrame({"Vùng": rn, "Expert": C_exp.round(4),
                            "Entropy": C_ent.round(4), "AHP": C_ahp.round(4)})
        st.dataframe(cmp, use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame({"Tiêu chí": TOPSIS_LBL, "AHP w": w_ahp.round(4)}),
                     use_container_width=True, hide_index=True)


# ============================================================
# BÀI 7 — NSGA-II PARETO
# ============================================================
BETA_MAT7 = np.array([[1.15, 0.85, 0.55, 1.30], [0.95, 1.25, 1.40, 1.05],
                      [1.05, 0.95, 0.85, 1.15], [1.20, 0.75, 0.45, 1.35],
                      [0.90, 1.30, 1.55, 1.00], [1.10, 0.85, 0.65, 1.25]])
D0_7 = np.array([38, 78, 55, 32, 82, 48])
E_7 = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
RHO_7 = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
SIG_7 = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])


@st.cache_data(show_spinner=False)
def _run_nsga(pop=100, gen=200):
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as moo_min
    from pymoo.termination import get_termination

    class P(ElementwiseProblem):
        def __init__(s):
            super().__init__(n_var=24, n_obj=4, n_ieq_constr=20,
                             xl=np.zeros(24), xu=np.ones(24) * 12000)

        def _evaluate(s, x, out, *a, **k):
            X = x.reshape(6, 4)
            f1 = -(BETA_MAT7 * X).sum()
            su = X.sum(1); f2 = np.abs(su - su.mean()).mean()
            f3 = (E_7 * (X[:, 0] + X[:, 1] + X[:, 2])).sum()
            f4 = (RHO_7 * X[:, 2]).sum() - (SIG_7 * X[:, 3]).sum()
            out['F'] = [f1, f2, f3, f4]
            g = [X.sum() - 50000]
            for r in range(6): g.append(5000 - X[r].sum())
            for r in range(6): g.append(X[r].sum() - 12000)
            g.append(12000 - X[:, 3].sum())
            Dn = D0_7 + 0.002 * X[:, 1]; Dm = Dn.max()
            for r in range(6): g.append(0.6 * Dm - Dn[r])
            out['G'] = np.array(g)

    res = moo_min(P(), NSGA2(pop_size=pop), get_termination("n_gen", gen),
                  seed=42, verbose=False, save_history=False)
    if res.F is not None and len(np.atleast_2d(res.F)) > 0:
        F = np.atleast_2d(res.F); X = np.atleast_2d(res.X)
    else:
        # Không có nghiệm khả thi tuyệt đối -> lấy quần thể cuối, sắp theo độ vi phạm
        pop_obj = res.pop
        Xall = pop_obj.get("X")
        Fall = pop_obj.get("F")
        Gall = pop_obj.get("G")
        cv = np.maximum(0, Gall).sum(axis=1)
        keep = np.argsort(cv)[:max(20, pop // 2)]
        F, X = Fall[keep], Xall[keep]
    return F, X


def page_bai7():
    section("🌐 Bài 7 — Tối ưu đa mục tiêu Pareto với NSGA-II",
            "4 mục tiêu xung đột: tăng trưởng · bao trùm · môi trường · an ninh dữ liệu")
    try:
        with st.spinner("Đang chạy NSGA-II (pop=80, gen=120)..."):
            F, X = _run_nsga()
    except ImportError:
        st.error("Cần cài pymoo: `pip install pymoo`")
        return

    # TOPSIS trên Pareto chọn nghiệm thỏa hiệp
    w_pol = np.array([0.40, 0.25, 0.20, 0.15])
    fmin, fmax = F.min(0), F.max(0)
    fr = np.where(fmax - fmin > 1e-12, fmax - fmin, 1.0)
    R = (F - fmin) / fr
    V = R * w_pol
    S_s = np.sqrt((V**2).sum(1))
    S_n = np.sqrt(((V - w_pol)**2).sum(1))
    Cs = S_n / (S_s + S_n)
    best = int(np.argmax(Cs))
    mg = int(np.argmin(F[:, 0]))

    c1, c2, c3 = st.columns(3)
    kpi(c1, "Số nghiệm Pareto", f"{len(F)}")
    kpi(c2, "GDP gain (thỏa hiệp)", f"{-F[best,0]:,.0f} tỷ")
    kpi(c3, "GDP gain (max)", f"{-F[mg,0]:,.0f} tỷ")

    tab1, tabp, tab2 = st.tabs(["7.4.2 Biên Pareto 3D", "Parallel coordinates",
                                "7.4.3-4 Nghiệm thỏa hiệp"])
    with tab1:
        fig = go.Figure(go.Scatter3d(
            x=-F[:, 0], y=F[:, 1], z=F[:, 2], mode="markers",
            marker=dict(size=4, color=F[:, 3], colorscale="Viridis",
                        colorbar=dict(title="Rủi ro"))))
        fig.add_trace(go.Scatter3d(x=[-F[best, 0]], y=[F[best, 1]], z=[F[best, 2]],
                                   mode="markers", marker=dict(size=9, color="red"),
                                   name="Thỏa hiệp"))
        fig.update_layout(template=PLOT_TMPL, height=560,
                          scene=dict(xaxis_title="GDP gain", yaxis_title="Gini/MAD",
                                     zaxis_title="Phát thải"),
                          title="Tập Pareto 3D (màu = rủi ro an ninh)")
        st.plotly_chart(fig, use_container_width=True)
    with tabp:
        fn = np.copy(F).astype(float)
        for i in range(4):
            lo, hi = F[:, i].min(), F[:, i].max()
            fn[:, i] = (F[:, i] - lo) / (hi - lo) if hi > lo else 0.5
        figp = go.Figure(go.Parcoords(
            line=dict(color=fn[:, 0], colorscale="Plasma"),
            dimensions=[
                dict(label="GDP gain (↓min)", values=fn[:, 0]),
                dict(label="Gini (↓min)", values=fn[:, 1]),
                dict(label="Phát thải (↓min)", values=fn[:, 2]),
                dict(label="Rủi ro (↓min)", values=fn[:, 3])]))
        figp.update_layout(template=PLOT_TMPL, height=460,
                           title="Parallel coordinates — 4 mục tiêu (chuẩn hóa [0,1])")
        st.plotly_chart(figp, use_container_width=True)
    with tab2:
        dfm = pd.DataFrame(bX, index=REGIONS_VI, columns=ITEMS).round(0)
        dfm["Tổng"] = dfm.sum(1)
        st.write("**Phân bổ nghiệm thỏa hiệp (TOPSIS, w=0.40/0.25/0.20/0.15):**")
        st.dataframe(dfm, use_container_width=True)
        oc = pd.DataFrame({
            "Mục tiêu": ["GDP gain", "Gini/MAD", "Phát thải", "Rủi ro"],
            "Thỏa hiệp": [-F[best, 0], F[best, 1], F[best, 2], F[best, 3]],
            "Tăng trưởng max": [-F[mg, 0], F[mg, 1], F[mg, 2], F[mg, 3]]}).round(1)
        st.write("**Chi phí cơ hội (thỏa hiệp vs tăng trưởng cao nhất):**")
        st.dataframe(oc, use_container_width=True, hide_index=True)


# ============================================================
# BÀI 8 — TỐI ƯU ĐỘNG 2026-2035
# ============================================================
@st.cache_data(show_spinner=False)
def _run_dynamic():
    from scipy.optimize import minimize
    a, b, g, d, th = 0.33, 0.42, 0.10, 0.08, 0.07
    dK, dD, dAI = 0.05, 0.12, 0.15
    thH, mu = 0.8, 0.02
    p1, p2, p3 = 0.003, 0.002, 0.004
    rho, gcr, T = 0.97, 1.5, 10
    K0, L0, D0, AI0, H0, Y0 = 27500., 53.9, 20.3, 86., 30., 12847.6
    A0 = Y0 / (K0**a * L0**b * D0**g * AI0**d * H0**th)
    L = np.array([L0 * 1.009**t for t in range(T + 1)])

    def traj(u, shock_year=None, shock_pct=0.0):
        IK, ID, IAI, IH = u[0::4], u[1::4], u[2::4], u[3::4]
        K = np.zeros(T + 1); D = np.zeros(T + 1); AI = np.zeros(T + 1)
        H = np.zeros(T + 1); A = np.zeros(T + 1); Y = np.zeros(T + 1); C = np.zeros(T)
        K[0], D[0], AI[0], H[0], A[0] = K0, D0, AI0, H0, A0
        for t in range(T):
            if shock_year is not None and t == shock_year:
                A[t] *= (1 - shock_pct)
            Y[t] = A[t] * K[t]**a * L[t]**b * D[t]**g * AI[t]**d * H[t]**th
            C[t] = Y[t] - IK[t] - ID[t] - IAI[t] - IH[t]
            if C[t] <= 0: return None
            K[t + 1] = (1 - dK) * K[t] + IK[t]
            D[t + 1] = (1 - dD) * D[t] + ID[t]
            AI[t + 1] = (1 - dAI) * AI[t] + IAI[t]
            H[t + 1] = H[t] + thH * IH[t] - mu * H[t]
            A[t + 1] = A[t] * (1 + p1 * D[t] / 100 + p2 * AI[t] / 100 + p3 * H[t] / 100)
        Y[T] = A[T] * K[T]**a * L[T]**b * D[T]**g * AI[T]**d * H[T]**th
        return K, D, AI, H, Y, C, A

    def welfare(u, sy=None, sp=0.0):
        r = traj(u, sy, sp)
        if r is None or np.any(r[5] <= 0): return 1e15
        C = r[5]
        return -sum(rho**t * (C[t]**(1 - gcr) - 1) / (1 - gcr) for t in range(T))

    ti = 14000 * 0.15
    u0 = np.tile([ti * 0.40, ti * 0.25, ti * 0.20, ti * 0.15], T)

    def cons_factory(sy=None, sp=0.0):
        def cons(u):
            r = traj(u, sy, sp)
            return -1e10 if r is None else min(r[5]) - 1
        return cons

    res = minimize(welfare, u0, method='SLSQP', bounds=[(0, None)] * (T * 4),
                   constraints=[{'type': 'ineq', 'fun': cons_factory()}],
                   options={'maxiter': 600, 'ftol': 1e-8})

    # 8.3.3 — sốc TFP 8% năm 2028 (t=2): giữ kế hoạch vs tái tối ưu
    W_base = -welfare(res.x)
    W_plan = -welfare(res.x, 2, 0.08)
    res_sh = minimize(lambda u: welfare(u, 2, 0.08), res.x, method='SLSQP',
                      bounds=[(0, None)] * (T * 4),
                      constraints=[{'type': 'ineq', 'fun': cons_factory(2, 0.08)}],
                      options={'maxiter': 600, 'ftol': 1e-8})
    W_reopt = -res_sh.fun
    Y_base = traj(res.x)[4]
    Y_shock = traj(res.x, 2, 0.08)[4]
    Y_reopt = traj(res_sh.x, 2, 0.08)[4]

    # 8.3.4 — đầu tư đều vs front-load
    u_even = u0.copy()
    u_front = np.zeros(T * 4)
    for t in range(T):
        f = 1.5 if t < 3 else 0.7
        u_front[t * 4:(t + 1) * 4] = np.array([ti * 0.40, ti * 0.25, ti * 0.20, ti * 0.15]) * f
    W_even, W_front = -welfare(u_even), -welfare(u_front)
    Y_even, Y_front = traj(u_even)[4], traj(u_front)[4]

    return {"opt": traj(res.x), "W": -res.fun,
            "shock": dict(W_base=W_base, W_plan=W_plan, W_reopt=W_reopt,
                          Y_base=Y_base, Y_shock=Y_shock, Y_reopt=Y_reopt),
            "strat": dict(W_opt=-res.fun, W_even=W_even, W_front=W_front,
                          Y_opt=traj(res.x)[4], Y_even=Y_even, Y_front=Y_front)}


def page_bai8():
    section("⏳ Bài 8 — Tối ưu động phân bổ liên thời gian 2026-2035",
            "Quỹ đạo tối ưu K, D, AI, H, Y, C · SLSQP · hàm thỏa dụng CRRA")
    with st.spinner("Đang tối ưu quỹ đạo 10 năm (SLSQP)..."):
        out = _run_dynamic()
    (K, D, AI, H, Y, C, A) = out["opt"]; W = out["W"]
    years = list(range(2026, 2037))
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Phúc lợi W*", f"{W:.2f}")
    kpi(c2, "GDP 2035", f"{Y[-1]:,.0f} ng.tỷ")
    kpi(c3, "Tăng trưởng BQ", f"{((Y[-1]/Y[0])**(1/10)-1)*100:.2f}%/năm")

    tab1, tab3, tab4 = st.tabs(["8.3.1-2 Quỹ đạo tối ưu",
                                "8.3.3 Phân tích cú sốc 2028", "8.3.4 So sánh chiến lược"])
    with tab1:
        fig = make_subplots(rows=2, cols=3, subplot_titles=(
            "K (vốn vật chất)", "D (hạ tầng số %)", "AI (nghìn DN)",
            "H (nhân lực %)", "Y & C", "A (TFP)"))
        series = [(K, 1, 1, "#60a5fa"), (D, 1, 2, "#22d3ee"), (AI, 1, 3, "#a78bfa"),
                  (H, 2, 1, "#fb923c"), (A, 2, 3, "#4ade80")]
        for s, r, c, col in series:
            fig.add_trace(go.Scatter(x=years, y=s, mode="lines+markers",
                                     line_color=col, showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(x=years, y=Y, name="Y", line_color="#f1f5f9"), row=2, col=2)
        fig.add_trace(go.Scatter(x=years[:10], y=C, name="C", line_color="#22d3ee"), row=2, col=2)
        fig.update_layout(template=PLOT_TMPL, height=620, title="Quỹ đạo tối ưu 2026-2035")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="note"><b>Thảo luận:</b> Quỹ đạo <b>front-load</b> đầu tư AI & H '
                    'giai đoạn đầu để tích lũy năng lực hấp thụ. Đào tạo nhân lực (H) nên đi '
                    'trước/đồng thời với đầu tư AI.</div>', unsafe_allow_html=True)
    with tab3:
        sh = out["shock"]
        cc = st.columns(3)
        kpi(cc[0], "W không sốc", f"{sh['W_base']:.2f}")
        kpi(cc[1], "W giữ kế hoạch", f"{sh['W_plan']:.2f}")
        kpi(cc[2], "W tái tối ưu", f"{sh['W_reopt']:.2f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=sh["Y_base"], name="Không sốc",
                                 line_color="#4ade80"))
        fig.add_trace(go.Scatter(x=years, y=sh["Y_shock"], name="Có sốc (giữ KH)",
                                 line_color="#f87171"))
        fig.add_trace(go.Scatter(x=years, y=sh["Y_reopt"], name="Có sốc (tái tối ưu)",
                                 line_color="#fbbf24", line=dict(dash="dot")))
        fig.add_vline(x=2028, line_dash="dash", line_color="#94a3b8")
        fig.update_layout(template=PLOT_TMPL, title="Cú sốc TFP -8% năm 2028 (như bão Yagi)",
                          xaxis_title="Năm", yaxis_title="GDP (ng.tỷ)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Sốc TFP persistent kéo dài qua động học A[t+1]=A[t]·(1+growth). "
                   "Tái tối ưu giúp phục hồi một phần phúc lợi.")
    with tab4:
        s = out["strat"]
        df = pd.DataFrame({
            "Chiến lược": ["Tối ưu (SLSQP)", "Đầu tư đều", "Front-load"],
            "Phúc lợi W": [round(s["W_opt"], 2), round(s["W_even"], 2), round(s["W_front"], 2)],
            "GDP 2035": [round(s["Y_opt"][-1]), round(s["Y_even"][-1]), round(s["Y_front"][-1])]})
        st.dataframe(df, use_container_width=True, hide_index=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=s["Y_opt"], name="Tối ưu", line_color="#f43f5e"))
        fig.add_trace(go.Scatter(x=years, y=s["Y_even"], name="Đều", line_color="#60a5fa"))
        fig.add_trace(go.Scatter(x=years, y=s["Y_front"], name="Front-load", line_color="#fb923c"))
        fig.update_layout(template=PLOT_TMPL, title="So sánh quỹ đạo GDP theo chiến lược",
                          xaxis_title="Năm", yaxis_title="GDP (ng.tỷ)")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# BÀI 9 — LAO ĐỘNG & AI
# ============================================================
def page_bai9():
    from scipy.optimize import linprog
    section("👷 Bài 9 — Tác động AI tới thị trường lao động Việt Nam",
            "Phân bổ 30.000 tỷ cho x_AI & x_H · tối đa hóa NetJob ròng 8 ngành")
    N = 8
    sec = ['Nông-LT', 'CN chế biến', 'Xây dựng', 'Bán buôn-bán lẻ',
           'Tài chính-NH', 'Logistics', 'CNTT-TT', 'Giáo dục-ĐT']
    L = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
    risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
    a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
    b1 = np.array([45, 28, 35, 32, 22, 30, 20, 55])
    c1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
    d1 = np.array([50, 32, 42, 38, 26, 36, 24, 62])
    coeff = a1 - c1 * risk

    def solve(cap5=False):
        c_obj = np.concatenate([-coeff, -b1])
        A1 = np.concatenate([np.ones(N), np.ones(N)]).reshape(1, -1)
        A1b = np.concatenate([-np.ones(N), np.zeros(N)]).reshape(1, -1)
        A2 = np.zeros((N, 2 * N)); A3 = np.zeros((N, 2 * N))
        for i in range(N):
            A2[i, i] = -coeff[i]; A2[i, N + i] = -b1[i]
            A3[i, i] = c1[i] * risk[i]; A3[i, N + i] = -d1[i]
        A_ub = np.vstack([A1, A1b, A2, A3])
        b_ub = np.concatenate([[30000], [-9000], np.zeros(N), np.zeros(N)])
        if cap5:
            A4 = np.zeros((N, 2 * N))
            for i in range(N): A4[i, i] = c1[i] * risk[i]
            A_ub = np.vstack([A_ub, A4]); b_ub = np.concatenate([b_ub, 0.05 * L * 1e6])
        return linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * (2 * N), method="highs")

    res = solve()
    xA, xH = res.x[:N], res.x[N:]
    NetJob = coeff * xA + b1 * xH
    Displaced = c1 * risk * xA
    c1c, c2c = st.columns(2)
    kpi(c1c, "Tổng NetJob ròng", f"{-res.fun:,.0f} việc")
    kpi(c2c, "Ngân sách dùng", f"{xA.sum()+xH.sum():,.0f} / 30.000 tỷ")

    tab1, tab2s, tab3s, tab2 = st.tabs(["9.4.1 Phân bổ & NetJob",
                                        "9.4.2 Ngưỡng đào tạo", "9.4.3 Luồng lao động (Sankey)",
                                        "9.4.4 Ràng buộc displaced ≤ 5%L"])
    with tab1:
        df = pd.DataFrame({"Ngành": sec, "x_AI": xA.round(0), "x_H": xH.round(0),
                           "Displaced": Displaced.round(0), "NetJob": NetJob.round(0)})
        st.dataframe(df, use_container_width=True, hide_index=True)
        fig = px.bar(df, x="Ngành", y="NetJob", template=PLOT_TMPL,
                     color="NetJob", color_continuous_scale="Greens",
                     title="NetJob ròng theo ngành")
        st.plotly_chart(fig, use_container_width=True)
    with tab2s:
        i = 1  # CN chế biến
        net = a1[i] - c1[i] * risk[i]
        rr = c1[i] * risk[i] / d1[i]
        st.write(f"**Ngành CN chế biến chế tạo:** hệ số net AI = {net:.1f}, "
                 f"tỷ lệ retrain = c₁·risk/d₁ = **{rr:.3f}**.")
        st.write(f"Ràng buộc: Displaced ≤ RetrainCapacity → x_H ≥ {rr:.3f}·x_AI. "
                 f"Mỗi 1 tỷ đầu tư AI cần ≈ {rr:.3f} tỷ đào tạo.")
        xr = np.linspace(0, 30000, 100)
        xh_re = rr * xr
        xh_nj = np.maximum(0, -net / b1[i] * xr)
        xh_min = np.maximum(xh_re, xh_nj)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xr, y=xh_re, name="Ngưỡng retrain", line_color="#f87171"))
        fig.add_trace(go.Scatter(x=xr, y=xh_nj, name="Ngưỡng NetJob≥0", line_color="#60a5fa"))
        fig.add_trace(go.Scatter(x=xr, y=xh_min, name="Khả thi (≥)", fill="tonexty",
                                 line_color="#4ade80", opacity=0.3))
        fig.update_layout(template=PLOT_TMPL, xaxis_title="x_AI (tỷ)",
                          yaxis_title="x_H tối thiểu (tỷ)",
                          title="Ngưỡng đào tạo tối thiểu — CN chế biến")
        st.plotly_chart(fig, use_container_width=True)
    with tab3s:
        vuln = [0, 2, 3]  # Nông-LT, Xây dựng, Bán buôn
        RetrainCap = d1 * xH
        labels, src, tgt, val, colr = [], [], [], [], []
        for k in vuln:
            labels.append(sec[k])
        labels += ["Giữ việc", "Đào tạo lại", "Mất việc"]
        for idx, k in enumerate(vuln):
            disp = Displaced[k]
            retr = min(disp, RetrainCap[k])
            lost = max(0, disp - retr)
            kept = L[k] * 1e6 - disp
            for tnode, v, col in [(len(vuln), kept, "#4ade80"),
                                  (len(vuln) + 1, retr, "#fbbf24"),
                                  (len(vuln) + 2, lost, "#f87171")]:
                if v > 0:
                    src.append(idx); tgt.append(tnode); val.append(v); colr.append(col)
        fig = go.Figure(go.Sankey(
            node=dict(label=labels, pad=18, thickness=18,
                      color=["#60a5fa"] * len(vuln) + ["#4ade80", "#fbbf24", "#f87171"]),
            link=dict(source=src, target=tgt, value=val, color=colr)))
        fig.update_layout(template=PLOT_TMPL, height=420,
                          title="Luồng dịch chuyển lao động nhóm dễ tổn thương")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        res4 = solve(cap5=True)
        if res4.success:
            st.success(f"Khả thi · NetJob = {-res4.fun:,.0f} "
                       f"(giảm {(-res.fun)-(-res4.fun):,.0f} = "
                       f"{((-res.fun)-(-res4.fun))/(-res.fun)*100:.1f}%)")
        else:
            st.error("KHÔNG khả thi với ràng buộc displaced ≤ 5% lao động.")
        st.markdown('<div class="note"><b>Ràng buộc cốt lõi:</b> Displaced ≤ RetrainingCapacity '
                    '(c₁·risk·x_AI ≤ d₁·x_H) — "tốc độ tự động hóa không vượt quá năng lực đào '
                    'tạo lại". Ngành CN chế biến & Tài chính cần đầu tư H nhiều nhất.</div>',
                    unsafe_allow_html=True)


# ============================================================
# BÀI 10 — STOCHASTIC SP
# ============================================================
J10 = ['I', 'D', 'AI', 'H']
S10 = ['s1', 's2', 's3', 's4']
P_S = {'s1': 0.30, 's2': 0.45, 's3': 0.20, 's4': 0.05}
BETA_BASE = {'I': 1.00, 'D': 1.10, 'AI': 1.25, 'H': 0.95}
BETA_S = {('s1', 'I'): 1.25, ('s1', 'D'): 1.35, ('s1', 'AI'): 1.55, ('s1', 'H'): 1.05,
          ('s2', 'I'): 1.00, ('s2', 'D'): 1.10, ('s2', 'AI'): 1.25, ('s2', 'H'): 0.95,
          ('s3', 'I'): 0.75, ('s3', 'D'): 0.85, ('s3', 'AI'): 0.90, ('s3', 'H'): 1.00,
          ('s4', 'I'): 0.40, ('s4', 'D'): 0.50, ('s4', 'AI'): 0.55, ('s4', 'H'): 1.10}


def _solve_sp():
    """Giải SP, EV, WS bằng scipy.linprog (không phụ thuộc Pyomo/solver ngoài)."""
    from scipy.optimize import linprog
    # Stochastic: biến [x_I,x_D,x_AI,x_H, y_s1(4), y_s2(4), y_s3(4), y_s4(4)] = 20 biến
    n = 4 + 16
    c = np.zeros(n)
    for k, j in enumerate(J10):
        c[k] = -BETA_BASE[j]
    for si, s in enumerate(S10):
        for k, j in enumerate(J10):
            c[4 + si * 4 + k] = -P_S[s] * BETA_S[(s, j)]
    A_ub = []; b_ub = []
    A_ub.append([1, 1, 1, 1] + [0] * 16); b_ub.append(65000)          # budget1
    for si in range(4):                                                # budget2 mỗi s
        row = [0] * n
        for k in range(4): row[4 + si * 4 + k] = 1
        A_ub.append(row); b_ub.append(15000)
    for si in range(4):                                                # y_AI <= 0.5 x_H
        row = [0] * n
        row[3] = -0.5; row[4 + si * 4 + 2] = 1
        A_ub.append(row); b_ub.append(0)
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=[(0, None)] * n, method="highs")
    x_sp = res.x[:4]; Z_sp = -res.fun
    y_sp = {s: res.x[4 + i * 4:4 + i * 4 + 4] for i, s in enumerate(S10)}

    # Deterministic mỗi scenario (Z*[s]) + WS
    det = {}
    for s in S10:
        cs = np.zeros(8)
        for k, j in enumerate(J10):
            cs[k] = -BETA_BASE[j]; cs[4 + k] = -BETA_S[(s, j)]
        Aub = [[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1],
               [0, 0, 0, -0.5, 0, 0, 1, 0]]
        bub = [65000, 15000, 0]
        r = linprog(cs, A_ub=np.array(Aub), b_ub=np.array(bub),
                    bounds=[(0, None)] * 8, method="highs")
        det[s] = {"Z": -r.fun, "x": r.x[:4]}
    Z_ws = sum(P_S[s] * det[s]["Z"] for s in S10)

    # EV: dùng beta trung bình cho first-stage
    beta_avg = {j: sum(P_S[s] * BETA_S[(s, j)] for s in S10) for j in J10}
    cev = [-beta_avg[j] for j in J10]
    rev = linprog(cev, A_ub=[[1, 1, 1, 1]], b_ub=[65000],
                  bounds=[(0, None)] * 4, method="highs")
    x_ev = rev.x
    Z_ev = sum(BETA_BASE[j] * x_ev[k] for k, j in enumerate(J10))
    for s in S10:                       # tối ưu y theo x_ev cố định
        cs = [-BETA_S[(s, j)] for j in J10]
        Aub = [[1, 1, 1, 1], [0, 0, 1, 0]]
        bub = [15000, 0.5 * x_ev[3]]
        r = linprog(cs, A_ub=np.array(Aub), b_ub=np.array(bub),
                    bounds=[(0, None)] * 4, method="highs")
        Z_ev += P_S[s] * (-r.fun)
    # 10.5.4 — Robust (minimax regret): min w, s.t. Z*[s]-(beta*x+beta_s*y_s)<=w
    # biến: [x(4), y_s1..s4(16), w] = 21
    nr = 21
    cr = np.zeros(nr); cr[20] = 1.0
    Ar = []; br = []
    rowb = [0] * nr
    for k in range(4): rowb[k] = 1
    Ar.append(rowb); br.append(65000)                      # budget1
    for si in range(4):
        row = [0] * nr
        for k in range(4): row[4 + si * 4 + k] = 1
        Ar.append(row); br.append(15000)                   # budget2
    for si in range(4):
        row = [0] * nr; row[3] = -0.5; row[4 + si * 4 + 2] = 1
        Ar.append(row); br.append(0)                       # y_AI<=0.5 x_H
    for si, s in enumerate(S10):                            # regret <= w
        row = [0] * nr
        for k, j in enumerate(J10):
            row[k] = -BETA_BASE[j]; row[4 + si * 4 + k] = -BETA_S[(s, j)]
        row[20] = -1
        Ar.append(row); br.append(-det[s]["Z"])
    rr = linprog(cr, A_ub=np.array(Ar), b_ub=np.array(br),
                 bounds=[(0, None)] * 20 + [(None, None)], method="highs")
    x_rob = rr.x[:4]; w_rob = rr.x[20]
    return x_sp, y_sp, Z_sp, Z_ev, Z_ws, det, x_rob, w_rob


def page_bai10():
    section("🎲 Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn",
            "First-stage (here-and-now) + recourse · VSS & EVPI · 4 kịch bản")
    x_sp, y_sp, Z_sp, Z_ev, Z_ws, det, x_rob, w_rob = _solve_sp()
    VSS = Z_sp - Z_ev
    EVPI = Z_ws - Z_sp

    c = st.columns(4)
    kpi(c[0], "Z* Stochastic", f"{Z_sp:,.0f}")
    kpi(c[1], "Z* EV solution", f"{Z_ev:,.0f}")
    kpi(c[2], "VSS", f"{VSS:,.0f}")
    kpi(c[3], "EVPI", f"{EVPI:,.0f}")

    tab1, tab2, tabr = st.tabs(["10.5.1 First & second stage",
                                "10.5.2-3 Kịch bản · VSS · EVPI",
                                "10.5.4 Robust (minimax regret)"])
    with tab1:
        st.write("**First-stage (here-and-now), tổng ≤ 65.000:**")
        st.dataframe(pd.DataFrame({"Hạng mục": J10, "x*": x_sp.round(0)}),
                     use_container_width=True, hide_index=True)
        ydf = pd.DataFrame({s: y_sp[s].round(0) for s in S10}, index=J10).T
        ydf.index = [f"{s} (p={P_S[s]})" for s in S10]
        st.write("**Second-stage recourse theo kịch bản (≤ 15.000):**")
        st.dataframe(ydf, use_container_width=True)
    with tab2:
        scn = pd.DataFrame({"Kịch bản": ["Lạc quan", "Cơ sở", "Bi quan", "Khủng hoảng"],
                            "Xác suất": [0.30, 0.45, 0.20, 0.05],
                            "Z*[s]": [round(det[s]["Z"], 0) for s in S10]})
        st.dataframe(scn, use_container_width=True, hide_index=True)
        fig = px.bar(scn, x="Kịch bản", y="Z*[s]", template=PLOT_TMPL,
                     color="Kịch bản", color_discrete_sequence=PALETTE,
                     title="Z* tối ưu theo từng kịch bản (deterministic)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'<div class="note"><b>VSS = {VSS:,.0f}</b>: giá trị của việc cân nhắc bất '
                    f'định khi ra quyết định. <b>EVPI = {EVPI:,.0f}</b>: giá trị thông tin hoàn '
                    f'hảo. SP đầu tư H (nhân lực) nhiều hơn EV — H là "hàng hóa bảo hiểm" giúp '
                    f'hấp thụ cú sốc.</div>', unsafe_allow_html=True)
    with tabr:
        st.write("Tối ưu robust cực tiểu hóa **regret kịch bản xấu nhất**: "
                 "min w, s.t. Z*[s] − Z(x,s) ≤ w ∀s.")
        kpi(st, "Minimax regret", f"{w_rob:,.0f}")
        cmp = pd.DataFrame({"Hạng mục": J10,
                            "x* Stochastic": x_sp.round(0),
                            "x* Robust": x_rob.round(0)})
        st.dataframe(cmp, use_container_width=True, hide_index=True)
        fig = go.Figure()
        fig.add_bar(x=J10, y=x_sp, name="Stochastic", marker_color="#60a5fa")
        fig.add_bar(x=J10, y=x_rob, name="Robust", marker_color="#fb923c")
        fig.update_layout(template=PLOT_TMPL, barmode="group",
                          title="First-stage: Stochastic vs Robust")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# BÀI 11 — Q-LEARNING
# ============================================================
class VietnamEconomyEnv:
    def __init__(self):
        self.allocation = {0: np.array([.70, .10, .10, .10]),
                           1: np.array([.40, .25, .15, .20]),
                           2: np.array([.25, .45, .15, .15]),
                           3: np.array([.20, .20, .45, .15]),
                           4: np.array([.30, .20, .10, .40])}
        self.action_names = ['Truyền thống', 'Cân bằng', 'Số hóa nhanh',
                             'AI dẫn dắt', 'Bao trùm']
        self.w = np.array([.40, .25, .20, .15]); self.T = 10
        self.rng = np.random.default_rng(0)

    def reset(self, state=None):
        self.state = np.array(state) if state is not None else self.rng.integers(0, 3, 4)
        self.t = 0; self.K, self.D, self.AI, self.H = 27500., 20.3, 86., 30.
        self.Y_prev = 12847.6
        return self.state.copy()

    def step(self, action):
        a = self.allocation[action]; budget = 2100.
        self.K = .95 * self.K + a[0] * budget
        self.D = .88 * self.D + a[1] * budget * .01
        self.AI = .85 * self.AI + a[2] * budget * .05
        self.H = self.H + .8 * a[3] * budget * .01 - .02 * self.H
        A = 33.70 * (1 + .003 * self.D / 100 + .002 * self.AI / 100 + .004 * self.H / 100)**self.t
        L = 53.9 * 1.009**self.t
        Y = A * self.K**.33 * L**.42 * self.D**.10 * self.AI**.08 * self.H**.07
        dg = (Y - self.Y_prev) / self.Y_prev
        du = max(0, -dg * .5); cy = self.AI / (self.H + 1) * .01
        em = (self.K + self.AI) * .0001
        r = self.w[0] * dg * 100 - self.w[1] * du * 100 - self.w[2] * cy - self.w[3] * em
        self.Y_prev = Y; self.t += 1
        gl = 0 if dg < .03 else (1 if dg < .06 else 2)
        dl = 0 if self.D < 25 else (1 if self.D < 35 else 2)
        al = 0 if self.AI < 100 else (1 if self.AI < 200 else 2)
        hl = 0 if self.H < 35 else (1 if self.H < 50 else 2)
        self.state = np.array([gl, dl, al, hl])
        return self.state.copy(), r, self.t >= self.T


@st.cache_data(show_spinner=False)
def _train_q(n_ep=8000):
    env = VietnamEconomyEnv()
    Q = np.zeros((3, 3, 3, 3, 5))
    hist = []
    for ep in range(n_ep):
        s = env.reset(); tot = 0; eps = max(.05, 1 - ep / 4000)
        while True:
            a = env.rng.integers(5) if env.rng.random() < eps else int(np.argmax(Q[tuple(s)]))
            s2, r, done = env.step(a)
            Q[tuple(s) + (a,)] += .1 * (r + .95 * Q[tuple(s2)].max() * (1 - done) - Q[tuple(s) + (a,)])
            tot += r; s = s2
            if done: break
        hist.append(tot)
    return Q, hist


def _eval_policy(Q, pol, n=300):
    env = VietnamEconomyEnv(); rs = []
    for _ in range(n):
        s = env.reset(); tot = 0
        while True:
            a = pol(s, Q, env); s, r, done = env.step(a); tot += r
            if done: break
        rs.append(tot)
    return np.mean(rs), np.std(rs)


def page_bai11():
    section("🤖 Bài 11 — Q-learning cho chính sách kinh tế thích nghi",
            "MDP 3⁴=81 trạng thái · 5 hành động · so sánh với rule-based")
    with st.spinner("Đang huấn luyện Q-learning (8.000 episodes)..."):
        Q, hist = _train_q()
    env = VietnamEconomyEnv()

    tab01, tab02, tab1, tab2 = st.tabs(
        ["11.3.1 Môi trường MDP", "11.3.2 Huấn luyện Q-learning",
         "11.3.3 Chính sách π*", "11.3.4 So sánh & learning curve"])
    with tab01:
        c1, c2 = st.columns(2)
        c1.markdown("**Không gian trạng thái** (3⁴ = 81):")
        c1.dataframe(pd.DataFrame({
            "Yếu tố": ["GDP growth", "Digital index", "AI capacity", "Unemploy risk"],
            "low (0)": ["<3%", "<25", "<100 ng.DN", "<35% H"],
            "med (1)": ["3-6%", "25-35", "100-200", "35-50%"],
            "high (2)": [">6%", ">35", ">200", ">50%"]}),
            hide_index=True, use_container_width=True)
        c2.markdown("**Không gian hành động** (5 lựa chọn % K/D/AI/H):")
        c2.dataframe(pd.DataFrame({
            "Hành động": env.action_names,
            "K": [70, 40, 25, 20, 30], "D": [10, 25, 45, 20, 20],
            "AI": [10, 15, 15, 45, 10], "H": [10, 20, 15, 15, 40]}),
            hide_index=True, use_container_width=True)
        st.markdown('<div class="note"><b>Phần thưởng:</b> R = 0.40·ΔGDP − 0.25·ΔUnemploy '
                    '− 0.20·CyberRisk − 0.15·Emission. Chuyển trạng thái mô phỏng bằng hàm sản '
                    'xuất Cobb-Douglas đã calibrate. Mỗi episode = 10 năm (T=10).</div>',
                    unsafe_allow_html=True)
    with tab02:
        c = st.columns(4)
        kpi(c[0], "Learning rate α", "0,10")
        kpi(c[1], "Discount γ", "0,95")
        kpi(c[2], "Episodes", "8.000")
        kpi(c[3], "Epsilon", "1,0 → 0,05")
        st.write("**Cập nhật Q-table** (Bellman, ε-greedy giảm dần):")
        st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha\,[\,r + \gamma \max_{a'}Q(s',a') - Q(s,a)\,]")
        w = 200
        sm = np.convolve(hist, np.ones(w) / w, mode="valid")
        fig = px.line(x=list(range(len(sm))), y=sm, template=PLOT_TMPL,
                      labels={"x": "Episode", "y": "Tổng phúc lợi (TB trượt 200)"},
                      title="Hội tụ phần thưởng trong quá trình huấn luyện")
        fig.update_traces(line_color="#22d3ee")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Q-table shape {Q.shape} · Q_max sau huấn luyện = {Q.max():.3f}")
    with tab1:
        tests = [([1, 1, 0, 1], "VN 2026 thực tế (GDP_med, D_med, AI_low, H_med)"),
                 ([0, 0, 0, 2], "Kịch bản tệ (GDP_low, D_low, AI_low, H_high)"),
                 ([2, 2, 2, 2], "Kịch bản tốt (GDP_high, D_high, AI_high, H_high)"),
                 ([0, 1, 0, 0], "Sau khủng hoảng (GDP_low, D_med, AI_low, H_low)"),
                 ([1, 0, 2, 1], "AI mạnh, D yếu (GDP_med, D_low, AI_high, H_med)")]
        rows = []
        for s, desc in tests:
            a = int(np.argmax(Q[tuple(s)]))
            rows.append({"Trạng thái khởi đầu": desc, "π* chọn": env.action_names[a]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with tab2:
        pols = [("π* (Q-learning)", lambda s, Q, e: int(np.argmax(Q[tuple(s)]))),
                ("Luôn Cân bằng (a1)", lambda s, Q, e: 1),
                ("Luôn AI dẫn dắt (a3)", lambda s, Q, e: 3),
                ("Random", lambda s, Q, e: e.rng.integers(5))]
        res = {n: _eval_policy(Q, p) for n, p in pols}
        dfp = pd.DataFrame({"Chính sách": list(res.keys()),
                            "Phúc lợi BQ": [round(v[0], 2) for v in res.values()],
                            "Std": [round(v[1], 2) for v in res.values()]})
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(dfp, use_container_width=True, hide_index=True)
            fig = px.bar(dfp, x="Chính sách", y="Phúc lợi BQ", template=PLOT_TMPL,
                         color="Chính sách", color_discrete_sequence=PALETTE)
            fig.update_layout(showlegend=False, title="So sánh chính sách")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            w = 200
            sm = np.convolve(hist, np.ones(w) / w, mode="valid")
            fig = px.line(x=list(range(len(sm))), y=sm, template=PLOT_TMPL,
                          labels={"x": "Episode", "y": "Tổng phúc lợi"},
                          title="Learning curve")
            fig.update_traces(line_color="#f43f5e")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="note"><b>Lưu ý:</b> AI hỗ trợ ra quyết định, '
                    '<b>không thay thế</b> trách nhiệm chính trị-xã hội. π* chỉ minh họa kỹ '
                    'thuật học chính sách thích nghi theo trạng thái kinh tế.</div>',
                    unsafe_allow_html=True)


# ============================================================
# BÀI 12 — AIDEOM-VN TÍCH HỢP (6 module / 4 tab)
# ============================================================
def _m1_forecast(alloc, T=4, budget=3000):
    a, b, g, d, th = 0.33, 0.42, 0.10, 0.08, 0.07
    K, D, AI, H, A, L0 = 27500., 20.3, 86., 30., 33.70, 53.9
    traj = [A * K**a * L0**b * D**g * AI**d * H**th]
    for t in range(T):
        K = .95 * K + alloc['K'] * budget
        D = .88 * D + alloc['D'] * budget * .01
        AI = .85 * AI + alloc['AI'] * budget * .05
        H = H + .8 * alloc['H'] * budget * .01 - .02 * H
        A = A * (1 + .003 * D / 100 + .002 * AI / 100 + .004 * H / 100)
        L = L0 * 1.009**(t + 1)
        traj.append(A * K**a * L**b * D**g * AI**d * H**th)
    return traj


M12_SCEN = {
    'S1 · Truyền thống': {'K': .70, 'D': .10, 'AI': .10, 'H': .10},
    'S2 · Số hóa nhanh': {'K': .25, 'D': .45, 'AI': .15, 'H': .15},
    'S3 · AI dẫn dắt': {'K': .20, 'D': .20, 'AI': .45, 'H': .15},
    'S4 · Bao trùm số': {'K': .30, 'D': .20, 'AI': .10, 'H': .40},
    'S5 · Tối ưu cân bằng': {'K': .25, 'D': .25, 'AI': .30, 'H': .20}}


def page_bai12():
    section("🇻🇳 Bài 12 — Nguyên mẫu AIDEOM-VN tích hợp",
            "6 module liên kết (M1→M6) · dashboard 4 tab · 5 kịch bản chính sách")

    years = list(range(2026, 2031))
    gdp_fc = {n: _m1_forecast(al) for n, al in M12_SCEN.items()}

    # M1 nhanh để hiện KPI đầu trang (giống ảnh bai12)
    a, b, g, d, th = 0.33, 0.42, 0.10, 0.08, 0.07
    Y = MACRO["GDP_trillion_VND"].values
    A = Y / (K_HIST**a * L_HIST**b * D_HIST**g * AI_HIST**d * H_HIST**th)
    Y_hat = A.mean() * (K_HIST**a * L_HIST**b * D_HIST**g * AI_HIST**d * H_HIST**th)
    mape = np.mean(np.abs((Y - Y_hat) / Y)) * 100
    Y2030 = gdp_fc['S5 · Tối ưu cân bằng'][-1]

    c1, c2, c3 = st.columns(3)
    kpi(c1, "MAPE (Cobb-Douglas)", f"{mape:.2f}%")
    kpi(c2, "Ā", f"{A.mean():.4f}")
    kpi(c3, "Y 2030 dự báo (S5)", f"{Y2030:,.0f} ng.tỷ")

    t1, t2, t3, t4 = st.tabs([
        "📊 Tổng quan (M1-M2)", "💰 Phân bổ (M3)",
        "🎬 5 Kịch bản (M6)", "⚠️ Cảnh báo rủi ro (M4-M5)"])

    # ---- TAB 1: M1 Dự báo + M2 TOPSIS ----
    with t1:
        st.markdown("#### M1 — Dự báo kinh tế (Cobb-Douglas)")
        gs = {"TFP (A)": (np.log(A[-1]) - np.log(A[0])) / 5,
              "Vốn (K)": a * (np.log(K_HIST[-1]) - np.log(K_HIST[0])) / 5,
              "Lao động (L)": b * (np.log(L_HIST[-1]) - np.log(L_HIST[0])) / 5,
              "Số hóa (D)": g * (np.log(D_HIST[-1]) - np.log(D_HIST[0])) / 5,
              "AI": d * (np.log(AI_HIST[-1]) - np.log(AI_HIST[0])) / 5,
              "Nhân lực số (H)": th * (np.log(H_HIST[-1]) - np.log(H_HIST[0])) / 5}
        dec = pd.DataFrame({"Yếu_tố": list(gs.keys()),
                            "đóng_góp_pct": [v * 100 for v in gs.values()]})
        fig = px.bar(dec, x="Yếu_tố", y="đóng_góp_pct", template=PLOT_TMPL,
                     color="Yếu_tố", color_discrete_sequence=PALETTE,
                     title="Phân rã đóng góp tăng trưởng 2020-2025")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### M2 — Đánh giá sẵn sàng số (TOPSIS)")
        X = REGIONS[TOPSIS_CRIT].values.astype(float)
        w_exp = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])
        C_exp = _topsis(X, w_exp, IS_BENEFIT)
        C_ent = _topsis(X, _entropy_w(X), IS_BENEFIT)
        m2 = pd.DataFrame({"Vùng": (REGIONS[REG_NAME_COL] if REG_NAME_COL else REGIONS_VI),
                           "C* Expert": C_exp.round(4), "C* Entropy": C_ent.round(4)})
        fig = go.Figure()
        fig.add_bar(y=m2["Vùng"], x=m2["C* Expert"], name="Expert",
                    orientation="h", marker_color="#60a5fa")
        fig.add_bar(y=m2["Vùng"], x=m2["C* Entropy"], name="Entropy",
                    orientation="h", marker_color="#fb923c")
        fig.update_layout(template=PLOT_TMPL, barmode="group",
                          title="TOPSIS Expert vs Entropy")
        st.plotly_chart(fig, use_container_width=True)

    # ---- TAB 2: M3 Phân bổ LP ----
    with t2:
        st.markdown("#### M3 — Tối ưu phân bổ ngành-vùng (LP, có C5 công bằng)")
        x_opt, Z_lp = _solve_lp4(True)
        kpi(st, "LP Z* (GDP gain)", f"{Z_lp:,.0f} tỷ VND")
        fig = px.imshow(x_opt, x=ITEMS, y=REGIONS_VI, aspect="auto", text_auto=".0f",
                        color_continuous_scale="YlOrRd", template=PLOT_TMPL,
                        title=f"Phân bổ tối ưu 6 vùng × 4 hạng mục (Z*={Z_lp:,.0f})")
        st.plotly_chart(fig, use_container_width=True)
        tot = x_opt.sum()
        st.caption(f"Tỷ lệ: D={x_opt[:,1].sum()/tot*100:.1f}% · "
                   f"AI={x_opt[:,2].sum()/tot*100:.1f}% · "
                   f"H={x_opt[:,3].sum()/tot*100:.1f}% → "
                   f"TFP boost động ước tính từ allocation.")

    # ---- TAB 3: M6 So sánh 5 kịch bản ----
    with t3:
        st.markdown("#### M6 — So sánh 5 kịch bản chính sách 2026-2030")
        fig = go.Figure()
        for (n, traj), col in zip(gdp_fc.items(), PALETTE):
            fig.add_trace(go.Scatter(x=years, y=traj, mode="lines+markers",
                                     name=n, line_color=col))
        fig.update_layout(template=PLOT_TMPL, title="Quỹ đạo GDP theo kịch bản",
                          xaxis_title="Năm", yaxis_title="GDP (ng.tỷ VND)")
        st.plotly_chart(fig, use_container_width=True)
        tbl = pd.DataFrame({
            "Kịch bản": list(gdp_fc.keys()),
            "GDP 2030": [round(t[-1], 0) for t in gdp_fc.values()],
            "TB %/năm": [round(((t[-1] / t[0])**(1 / 4) - 1) * 100, 2)
                         for t in gdp_fc.values()]})
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ---- TAB 4: M4 lao động + M5 rủi ro ----
    with t4:
        from scipy.optimize import linprog
        st.markdown("#### M4 — Mô phỏng thị trường lao động (NetJob)")
        N = 8
        sec = ['Nông-LT', 'CN chế biến', 'Xây dựng', 'Bán buôn', 'Tài chính',
               'Logistics', 'CNTT', 'Giáo dục']
        a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
        b1 = np.array([45, 28, 35, 32, 22, 30, 20, 55])
        c1v = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
        d1 = np.array([50, 32, 42, 38, 26, 36, 24, 62])
        risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
        coeff = a1 - c1v * risk
        c_obj = np.concatenate([-coeff, -b1])
        A1 = np.concatenate([np.ones(N), np.ones(N)]).reshape(1, -1)
        A2 = np.zeros((N, 2 * N)); A3 = np.zeros((N, 2 * N))
        for i in range(N):
            A2[i, i] = -coeff[i]; A2[i, N + i] = -b1[i]
            A3[i, i] = c1v[i] * risk[i]; A3[i, N + i] = -d1[i]
        res = linprog(c_obj, A_ub=np.vstack([A1, A2, A3]),
                      b_ub=np.concatenate([[30000], np.zeros(N), np.zeros(N)]),
                      bounds=[(0, None)] * (2 * N), method="highs")
        NJ = coeff * res.x[:N] + b1 * res.x[N:]
        kpi(st, "Tổng NetJob ròng", f"{-res.fun:,.0f} việc")
        fig = px.bar(x=sec, y=NJ.round(0), template=PLOT_TMPL,
                     color=NJ, color_continuous_scale="Greens",
                     labels={"x": "Ngành", "y": "NetJob"}, title="NetJob theo ngành")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### M5 — Đánh giá rủi ro (Stochastic theo kịch bản)")
        _, _, Z_sp, Z_ev, Z_ws, det, _, _ = _solve_sp()
        c1, c2, c3 = st.columns(3)
        kpi(c1, "Z* Stochastic", f"{Z_sp:,.0f}")
        kpi(c2, "VSS", f"{Z_sp - Z_ev:,.0f}")
        kpi(c3, "EVPI", f"{Z_ws - Z_sp:,.0f}")
        scn = pd.DataFrame({"Kịch bản": ["Lạc quan", "Cơ sở", "Bi quan", "Khủng hoảng"],
                            "Xác suất": [0.30, 0.45, 0.20, 0.05],
                            "Z*[s]": [round(det[s]["Z"], 0) for s in S10]})
        fig = px.bar(scn, x="Kịch bản", y="Z*[s]", template=PLOT_TMPL, color="Kịch bản",
                     color_discrete_sequence=['#4ade80', '#60a5fa', '#fb923c', '#f87171'],
                     title=f"Z* theo kịch bản · Z*_SP = {Z_sp:,.0f}")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="note"><b>Khuyến nghị tích hợp:</b> Kịch bản <b>S5 (cân bằng)</b> '
                    'cho GDP 2030 cao trong khi giữ công bằng vùng và NetJob dương. AIDEOM-VN '
                    'chỉ là công cụ hỗ trợ — quyết định cuối cùng thuộc về quy trình chính sách.'
                    '</div>', unsafe_allow_html=True)


# ============================================================
# ROUTER
# ============================================================
ROUTES = {
    PAGES[0]: page_home, PAGES[1]: page_bai1, PAGES[2]: page_bai2,
    PAGES[3]: page_bai3, PAGES[4]: page_bai4, PAGES[5]: page_bai5,
    PAGES[6]: page_bai6, PAGES[7]: page_bai7, PAGES[8]: page_bai8,
    PAGES[9]: page_bai9, PAGES[10]: page_bai10, PAGES[11]: page_bai11,
    PAGES[12]: page_bai12,
}
ROUTES[page]()

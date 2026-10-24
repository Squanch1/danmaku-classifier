import streamlit as st
import requests
import json
import time

# ---------------------------- 页面配置 ----------------------------
st.set_page_config(
    page_title="智能预测系统 · 科技版",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------- 自定义 CSS（科技感） ----------------------------
st.markdown("""
<style>
    /* 全局暗色背景与字体 */
    .stApp {
        background: linear-gradient(145deg, #0b0e1a 0%, #1a1f2f 100%);
        color: #e0e8f0;
        font-family: 'Segoe UI', 'Consolas', monospace;
    }
    /* 标题样式 */
    .main-title {
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7b2ffc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        letter-spacing: 2px;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #8892b0;
        font-size: 1.1rem;
        border-left: 3px solid #00d4ff;
        padding-left: 15px;
        margin-bottom: 2rem;
    }
    /* 输入框卡片 */
    .input-card {
        background: rgba(20, 28, 48, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid #2a3a5c;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        transition: border 0.3s ease;
    }
    .input-card:hover {
        border-color: #00d4ff;
    }
    /* 文本区域定制 */
    .stTextArea textarea {
        background: #111827 !important;
        color: #e0e8f0 !important;
        border: 1px solid #2a3a5c !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
        transition: all 0.2s;
    }
    .stTextArea textarea:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 12px rgba(0, 212, 255, 0.2) !important;
    }
    /* 按钮样式 */
    .stButton button {
        background: linear-gradient(135deg, #00d4ff, #7b2ffc) !important;
        color: white !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 0.6rem 2.5rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        width: 100%;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.6);
    }
    .stButton button:active {
        transform: scale(0.98);
    }
    /* 结果卡片 */
    .result-card {
        background: rgba(20, 28, 48, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid #2a3a5c;
        border-radius: 16px;
        padding: 1.2rem 1.8rem;
        margin-top: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border-left: 4px solid #00d4ff;
    }
    .result-label {
        color: #8892b0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .result-content {
        color: #ccd6f6;
        font-size: 1.1rem;
        word-wrap: break-word;
        white-space: pre-wrap;
        font-family: 'Consolas', monospace;
        background: #0f1625;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #1f2a44;
    }
    .metric-box {
        background: rgba(0, 212, 255, 0.08);
        border: 1px solid #1f2a44;
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 600;
        color: #00d4ff;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    /* 页脚 */
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #495670;
        font-size: 0.8rem;
        border-top: 1px solid #1f2a44;
        padding-top: 1.5rem;
    }
    /* 响应式修正 */
    @media (max-width: 768px) {
        .main-title { font-size: 2.2rem; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------- 页面头部 ----------------------------
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.markdown('<div class="main-title">⚡ 智能预测系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">基于深度学习的文本分析引擎 · 实时推理</div>', unsafe_allow_html=True)

# ---------------------------- 主布局 ----------------------------
left, right = st.columns([3, 1], gap="large")

with left:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    # 输入区
    input_text = st.text_area(
        "📝 输入待预测文本",
        value="中国人民公安大学2012年硕士研究生目录及书目",
        height=180,
        placeholder="请输入需要分析的文本内容...",
        key="input_text"
    )
    # 按钮（占据整行）
    predict_btn = st.button("🚀 启动预测", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 结果展示区
    if predict_btn and input_text.strip():
        with st.spinner("🧠 模型推理中，请稍候..."):
            try:
                start_time = time.time()
                api_url = "http://192.168.11.110:8000/predict"  # 请根据实际情况修改
                payload = {"text": input_text}
                response = requests.post(api_url, json=payload, timeout=30)
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    result_data = response.json()
                    # 将结果格式化输出
                    pretty_result = json.dumps(result_data, indent=2, ensure_ascii=False)

                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-label">📊 预测结果</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-content">{pretty_result}</div>', unsafe_allow_html=True)

                    # 显示响应时间指标
                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    with col_metric1:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value">{elapsed:.3f}s</div>
                            <div class="metric-label">响应时间</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_metric2:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value">{len(input_text)}</div>
                            <div class="metric-label">文本长度</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_metric3:
                        status_color = "#00d4ff" if response.status_code == 200 else "#ff6b6b"
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color:{status_color};">{response.status_code}</div>
                            <div class="metric-label">HTTP 状态</div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error(f"❌ 请求失败 (HTTP {response.status_code})：{response.text}")

            except requests.exceptions.ConnectionError:
                st.error("🚫 无法连接到服务器，请检查服务是否启动或地址是否正确。")
            except requests.exceptions.Timeout:
                st.error("⏰ 请求超时，请稍后重试。")
            except Exception as e:
                st.error(f"⚠️ 发生未知错误：{str(e)}")

    elif predict_btn and not input_text.strip():
        st.warning("✋ 请输入待预测的文本内容。")

with right:
    # 侧边信息卡片（科技感装饰）
    st.markdown("""
    <div style="background: rgba(20,28,48,0.5); backdrop-filter: blur(6px); border: 1px solid #2a3a5c; border-radius: 16px; padding: 1.2rem; margin-top: 0.5rem;">
        <div style="color: #8892b0; font-size: 0.8rem; letter-spacing: 1px;">⚙️ 系统状态</div>
        <div style="display: flex; align-items: center; gap: 8px; margin-top: 8px;">
            <span style="display: inline-block; width: 10px; height: 10px; background: #00d4ff; border-radius: 50%; box-shadow: 0 0 12px #00d4ff;"></span>
            <span style="color: #ccd6f6;">服务在线</span>
        </div>
        <div style="margin-top: 12px; color: #8892b0; font-size: 0.8rem;">
            <div>🔗 接口地址</div>
            <div style="color: #64ffda; word-break: break-all; font-size: 0.7rem; background: #0f1625; padding: 6px 10px; border-radius: 6px; margin-top: 4px;">http://192.168.11.229:8000/predict</div>
        </div>
        <div style="margin-top: 12px; color: #8892b0; font-size: 0.8rem;">
            <div>📦 模型版本</div>
            <div style="color: #ccd6f6;">v2.1.0</div>
        </div>
        <hr style="border-color: #1f2a44; margin: 16px 0;">
        <div style="color: #495670; font-size: 0.7rem; text-align: center;">
            <span>🔒 数据传输加密</span><br>
            <span>⚡ 异步高并发</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 增加一个小彩蛋——动态时钟（只是装饰）
    st.markdown("""
    <div style="margin-top: 1rem; background: rgba(20,28,48,0.3); border-radius: 10px; padding: 0.8rem; text-align: center; border: 1px solid #1f2a44;">
        <span style="color: #495670; font-size: 0.7rem;">🕒 当前系统时间</span><br>
        <span id="clock" style="color: #00d4ff; font-family: 'Consolas', monospace; font-size: 1.2rem;"></span>
    </div>
    <script>
        function updateClock() {
            const now = new Date();
            document.getElementById('clock').innerText = now.toLocaleTimeString('zh-CN', { hour12: false });
        }
        updateClock();
        setInterval(updateClock, 1000);
    </script>
    """, unsafe_allow_html=True)

# ---------------------------- 页脚 ----------------------------
st.markdown("""
<div class="footer">
    <span>© 2026 智能预测系统 · 基于 FastAPI + Streamlit 构建</span>
    <span style="margin: 0 12px;">|</span>
    <span>🚀 科技感交互 · 实时推理</span>
</div>
""", unsafe_allow_html=True)
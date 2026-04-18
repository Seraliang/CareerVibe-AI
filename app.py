"""
CareerVibe — 求职助手（Streamlit）
简历：PDF / .docx；JD：PDF 或文本。深度分析 + Plotly 五维雷达图。
"""

from __future__ import annotations

import html
import io
import json
import os
from pathlib import Path

import pdfplumber
import plotly.graph_objects as go
import streamlit as st
from docx import Document
from dotenv import load_dotenv
from openai import OpenAI

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv()

RADAR_LABELS = ["硬技能", "软实力", "行业相关度", "经验资历", "潜力"]

SYSTEM_HEADHUNTER = """你是具备全球化视野的专业全行业资深猎头顾问，熟悉互联网、金融、数据分析、产品与技术、运营与增长、创意与内容、专业服务、制造与供应链等多赛道与跨文化职场环境。
请先根据用户提供的岗位 JD 自行识别其所属细分行业与职能类型，不要默认局限于某一垂直；所有判断须与 JD 实际语境一致。
你的风格：像顶级咨询公司顾问撰写客户报告——结构清晰、论证充分、可执行；拒绝空话与过度精简；若材料不足须明确假设边界。"""


def inject_report_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stAppViewContainer"] {
            background: #f8f9fa !important;
        }
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            max-width: 1100px;
        }
        .hero-title {
            font-size: 2.15rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            background: linear-gradient(105deg, #0f172a 0%, #1e293b 40%, #334155 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.25rem;
        }
        .hero-sub {
            color: #64748b;
            font-size: 0.95rem;
            margin-bottom: 1.25rem;
        }
        .report-card {
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 24px rgba(15, 23, 42, 0.08);
            border: 1px solid rgba(30, 41, 59, 0.08);
            padding: 1.35rem 1.5rem;
            margin: 1rem 0;
        }
        .glass-panel {
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(8px);
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(30, 41, 59, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.8);
            padding: 1rem 1.25rem;
            margin: 0.75rem 0;
        }
        .career-path-card {
            border: 2px solid #c4b5fd;
            border-radius: 14px;
            padding: 1.35rem 1.5rem;
            margin: 1rem 0;
            background: linear-gradient(145deg, rgba(245, 243, 255, 0.95) 0%, #ffffff 55%);
            box-shadow: 0 6px 28px rgba(139, 92, 246, 0.12);
        }
        .career-path-card h3 {
            color: #5b21b6;
            margin-top: 0;
            font-size: 1.15rem;
        }
        [data-testid="stSidebar"] {
            background: #f1f5f9 !important;
            border-right: 1px solid #e2e8f0 !important;
        }
        [data-testid="stSidebar"] .stMarkdown { color: #475569; }
        .stTabs [data-baseweb="tab-list"] { gap: 10px; background: transparent; border-bottom: 1px solid #e2e8f0; }
        .stTabs [data-baseweb="tab"] {
            background: #fff;
            border: 1px solid #e2e8f0 !important;
            border-radius: 10px 10px 0 0 !important;
            color: #64748b;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background: #1e293b !important;
            border-color: #1e293b !important;
            color: #f8fafc !important;
        }
        h2, h3 { color: #1e293b !important; font-weight: 600 !important; }
        hr.soft { border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def extract_pdf_text(uploaded_file) -> str:
    raw = uploaded_file.getvalue()
    parts: list[str] = []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t.strip())
    return "\n\n".join(parts).strip()


def extract_docx_text(uploaded_file) -> str:
    doc = Document(io.BytesIO(uploaded_file.getvalue()))
    parts: list[str] = []
    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text.strip())
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts).strip()


def extract_resume_text(uploaded_file) -> str:
    name = (getattr(uploaded_file, "name", None) or "").lower()
    if name.endswith(".docx"):
        return extract_docx_text(uploaded_file)
    return extract_pdf_text(uploaded_file)


def get_api_key() -> str:
    return (
        os.getenv("DEEPSEEK_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )


def get_base_url() -> str:
    raw = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    return raw or "https://api.deepseek.com"


def get_client() -> OpenAI:
    key = get_api_key()
    if not key:
        st.error(
            "未检测到 API 密钥。请在 `.env` 中设置 `DEEPSEEK_API_KEY`（或临时兼容 `OPENAI_API_KEY`）。"
        )
        st.stop()
    return OpenAI(api_key=key, base_url=get_base_url())


def get_model() -> str:
    return (
        os.getenv("DEEPSEEK_MODEL", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
        or "deepseek-chat"
    )


def chat_completion(
    client: OpenAI,
    system: str,
    user: str,
    *,
    json_mode: bool = False,
    max_tokens: int = 4096,
) -> str:
    kwargs: dict = {
        "model": get_model(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.35,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def build_radar_figure(scores: dict) -> go.Figure:
    values = [int(scores.get(k, 0) or 0) for k in RADAR_LABELS]
    values = [max(0, min(100, v)) for v in values]
    theta = RADAR_LABELS + [RADAR_LABELS[0]]
    r = values + [values[0]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=r,
            theta=theta,
            fill="toself",
            fillcolor="rgba(30, 41, 59, 0.38)",
            line=dict(color="#1e293b", width=3.5),
            name="匹配维度",
            hovertemplate="%{theta}: %{r}<extra></extra>",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(30,41,59,0.15)"),
            angularaxis=dict(linecolor="rgba(30,41,59,0.2)"),
            bgcolor="rgba(255,255,255,0.4)",
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=48, b=48, l=48, r=48),
        font=dict(color="#1e293b", size=13, family="sans-serif"),
        title=dict(
            text="简历 × JD 五维匹配雷达",
            font=dict(size=16, color="#1e293b"),
            x=0.5,
        ),
    )
    return fig


def ensure_session_state() -> None:
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""
    if "jd_text" not in st.session_state:
        st.session_state.jd_text = ""


def main() -> None:
    st.set_page_config(
        page_title="CareerVibe",
        page_icon="\U0001f916",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_report_css()
    ensure_session_state()

    st.markdown(
        '<p class="hero-title">\U0001f916 CareerVibe · AI 职业战略报告</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-sub">Ultra-clean咨询报告风格 · 深度岗位画像 · 五维雷达 · 专属进路</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### 材料上传")
        st.markdown("**简历**：**PDF** 或 **Word（.docx）**。")
        resume_file = st.file_uploader(
            "简历文件",
            type=["pdf", "docx"],
            help="上传 PDF 或 .docx",
            key="resume_file",
        )
        st.markdown("**岗位 JD**")
        st.caption("有 PDF 时优先文件；否则使用下方文本。")
        jd_file = st.file_uploader("JD 文件（PDF，可选）", type=["pdf"], key="jd_pdf")
        jd_paste = st.text_area(
            "或直接粘贴 JD 全文",
            height=160,
            placeholder="未上传 JD PDF 时，在此粘贴岗位描述…",
            key="jd_paste",
        )

        if resume_file is None:
            st.session_state.resume_text = ""
        else:
            try:
                st.session_state.resume_text = extract_resume_text(resume_file)
            except Exception as e:  # noqa: BLE001
                st.error(f"简历解析失败：{e}")
                st.session_state.resume_text = ""

        if jd_file is not None:
            try:
                st.session_state.jd_text = extract_pdf_text(jd_file)
            except Exception as e:  # noqa: BLE001
                st.error(f"JD 文件解析失败：{e}")
                st.session_state.jd_text = ""
        elif jd_paste.strip():
            st.session_state.jd_text = jd_paste.strip()
        else:
            st.session_state.jd_text = ""

        st.divider()
        st.markdown("**解析状态**")
        r_ok = len(st.session_state.resume_text) > 50
        j_ok = len(st.session_state.jd_text) > 50
        jd_src = (
            "PDF"
            if jd_file is not None
            else ("文本" if jd_paste.strip() else "—")
        )
        st.write("简历：" + ("已读取" if r_ok else "— 待上传或内容过短"))
        st.write(
            "JD："
            + (
                "已读取（" + jd_src + "）"
                if j_ok
                else "— 请上传 PDF 或填写文本"
            )
        )

    client = get_client()
    materials_ok = len(st.session_state.resume_text) > 50 and len(
        st.session_state.jd_text
    ) > 50

    deep_report = st.session_state.get("deep_report")
    if isinstance(deep_report, dict) and materials_ok:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.35])
        with c1:
            overall = deep_report.get("overall_match_score")
            st.metric("综合匹配指数", f"{overall} / 100" if overall is not None else "—")
            st.caption("由模型综合五维与岗位语境估算，仅供参考。")
        with c2:
            rs = deep_report.get("radar_scores") or {}
            st.plotly_chart(build_radar_figure(rs), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

    tab_analysis, tab_interview, tab_debrief = st.tabs(
        ["深度岗位分析", "面试通关指南", "面试复盘评分"]
    )

    with tab_analysis:
        if not materials_ok:
            st.info("请先在侧边栏上传简历并填写/上传 JD（文本需达到一定长度）。")
        else:
            if st.button("生成完整深度分析报告", type="primary", key="btn_deep"):
                with st.spinner("正在生成结构化深度报告（约需 30–90 秒）…"):
                    user_prompt = f"""你将对以下「简历」与「岗位 JD」输出 **一份 JSON**，键名必须 **完全一致**（英文 snake_case），且 **仅输出 JSON**，不要 Markdown 代码围栏。

## 必须包含的键

1) overall_match_score：整数 0-100，表示简历与该 JD 的综合匹配度。

2) radar_scores：对象，**必须且仅含**这五个键（中文），值为0-100 的整数：
   "硬技能", "软实力", "行业相关度", "经验资历", "潜力"
   请严格依据简历与 JD 逐项独立打分并自洽。

3) job_portrait_md：字符串，使用 Markdown。**极尽详尽**，像咨询报告「岗位深度画像」章节，须覆盖：
   - 岗位在行业内的**稀缺度**、人才供需；在**公司业务链**中的位置与核心价值。
   - **职责深度拆解**：逐条或分块写出 JD 中的关键职责；对每一项写出**隐性能力要求**、常见 **KPI / 压力点**、踩坑点。
   - **薪资与前景透视**：结合你对 **2024–2026** 年该行业/城市/职级的公开市场认知，给出 **Base** 与 **Bonus**（若有）的**合理预估区间**，并明确你的**地域、公司类型、职级假设**；分析 **3–5 年** 典型**职业跃迁路径**（可含横向/纵向）。
   - 全文禁止敷衍或只列小标题不写实质段落。

4) interviewer_deep_questions：字符串数组，**恰好 3 条**。从**面试官视角**，写出最可能追问的**深度**问题（需体现业务/技术/行为深挖，而非泛泛「自我介绍」）。

5) gap_diagnosis_md：字符串，**纯深度叙述2–3 段**（不要用表格）。直指简历与 JD 之间的**核心断层**：能力、经历叙事、行业语境、证据链等；语气冷静、犀利、可执行。

6) action_plan_md：字符串，Markdown。**行动指南**：包含   - **简历话术升级**：**1–2 条**可直接改写进简历的句子或 bullet 建议（给 before/after 或可直接粘贴版本）。
   - **技能补强清单**：最紧迫要补的**技能 / 证书 / 项目关键词**，并说明为何与 JD 挂钩。

7) career_path_suggestion：对象，键为：
   - recommended_role：字符串，**推荐岗位名称**（可含级别，如「高级XX」「XX方向」）。
   - rationale：字符串，**详尽**说明为何该方向**最适配当前简历**（引用简历中的可验证亮点）。
   - market_competitiveness：字符串，分析该方向在**劳动力市场**的竞争格局、门槛与机会。
   **重要**：该对象须**主要基于「简历」推演最适配路径，可脱离当前 JD 的单一岗位叙事**；若与当前 JD 一致可点明，但**不得**被 JD 绑架成唯一答案。

--- 简历 ---
{st.session_state.resume_text[:10000]}

--- 岗位 JD ---
{st.session_state.jd_text[:10000]}
"""
                    raw = chat_completion(
                        client,
                        SYSTEM_HEADHUNTER,
                        user_prompt,
                        json_mode=True,
                        max_tokens=8192,
                    )
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        st.error("模型返回非标准 JSON，请重试或换模型。")
                        st.code(raw[:8000])
                    else:
                        st.session_state["deep_report"] = data

            deep_report = st.session_state.get("deep_report")
            if isinstance(deep_report, dict):
                t1, t2, t3 = st.tabs(
                    ["岗位深度画像", "核心差距与行动", "\U0001f680 专属职业进路"]
                )
                with t1:
                    st.markdown(
                        '<div class="report-card">', unsafe_allow_html=True
                    )
                    st.markdown(deep_report.get("job_portrait_md") or "—")
                    st.markdown("</div>", unsafe_allow_html=True)
                    qs = deep_report.get("interviewer_deep_questions") or []
                    if isinstance(qs, list) and qs:
                        st.markdown("##### 面试官视角 · 三大深度追问")
                        for i, q in enumerate(qs[:3], 1):
                            st.markdown(f"**{i}.** {q}")
                with t2:
                    st.markdown(
                        '<div class="report-card">', unsafe_allow_html=True
                    )
                    st.markdown("#### 核心差距诊断")
                    st.markdown(deep_report.get("gap_diagnosis_md") or "—")
                    st.markdown("---")
                    st.markdown("#### 行动指南（Action Plan）")
                    st.markdown(deep_report.get("action_plan_md") or "—")
                    st.markdown("</div>", unsafe_allow_html=True)
                with t3:
                    cps = deep_report.get("career_path_suggestion") or {}
                    if isinstance(cps, dict):
                        role = html.escape(str(cps.get("recommended_role") or "—"))
                        rat = html.escape(str(cps.get("rationale") or "—"))
                        mkt = html.escape(
                            str(cps.get("market_competitiveness") or "—")
                        )
                        st.markdown(
                            f"""
<div class="career-path-card">
<h3>\U0001f680 最适配岗位建议</h3>
<p><strong>推荐方向：</strong>{role}</p>
<p><strong>推荐理由</strong></p>
<p style="white-space:pre-wrap;">{rat}</p>
<p><strong>市场竞争力分析</strong></p>
<p style="white-space:pre-wrap;">{mkt}</p>
</div>
""",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.write("—")

    with tab_interview:
        st.subheader("面试通关指南")
        if not materials_ok:
            st.info("请先完成侧边栏简历与 JD。")
        else:
            if st.button("生成 20 题面试指南", key="btn_interview"):
                with st.spinner("正在生成高频问题与答案…"):
                    user_prompt = f"""根据「岗位 JD」识别行业与岗位类型，生成 **恰好 20 个** 高频面试问题；须与 JD 语境一致。
覆盖：硬技能与工具、业务场景、数据与结果、协作与沟通、领导力/影响力、压力与冲突、合规（如适用）、职业规划等。

对每个 i=1..20，输出：
**Q{{i}}** 问题
**简历深挖**：结合简历可核验事实的追问点
**建议回答**：详细个性化答案（第一人称，STAR 更佳）

Markdown，Q1…Q20编号完整。

--- 简历 ---
{st.session_state.resume_text[:10000]}

--- 岗位 JD ---
{st.session_state.jd_text[:10000]}
"""
                    out = chat_completion(
                        client,
                        SYSTEM_HEADHUNTER,
                        user_prompt,
                        json_mode=False,
                        max_tokens=8192,
                    )
                    st.session_state["interview_guide"] = out
            guide = st.session_state.get("interview_guide")
            if guide:
                st.markdown('<div class="report-card">', unsafe_allow_html=True)
                st.markdown(guide)
                st.markdown("</div>", unsafe_allow_html=True)

    with tab_debrief:
        st.subheader("面试复盘评分")
        notes = st.text_area(
            "粘贴面试复盘或转写",
            height=240,
            placeholder="题目、回答要点、反馈、自我评估…",
            key="debrief_notes",
        )
        if st.button("提交复盘并评分", key="btn_debrief"):
            if len(notes.strip()) < 30:
                st.warning("内容过短。")
            else:
                with st.spinner("评分中…"):
                    user_prompt = f"""以全行业资深猎头视角评估以下面试复盘。Markdown 输出：

## 百分制评分
整数0-100，2-4 句说明加减分。

## 优点总结
3-8 条。

## 详细改进建议
条列所有改进点；每条：问题表现 → 改进方向 → 可练习动作。

--- 面试复盘 ---
{notes[:14000]}
"""
                    out = chat_completion(
                        client,
                        SYSTEM_HEADHUNTER,
                        user_prompt,
                        json_mode=False,
                        max_tokens=4096,
                    )
                    st.session_state["debrief_result"] = out
        debrief = st.session_state.get("debrief_result")
        if debrief:
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.markdown(debrief)
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

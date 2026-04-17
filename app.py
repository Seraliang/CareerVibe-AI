"""
CareerVibe — 求职助手（Streamlit）
简历：pdfplumber（PDF）与 python-docx（.docx）；JD 支持 PDF 或文本粘贴。
分析通过 OpenAI 兼容 SDK 调用 DeepSeek API。
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pdfplumber
import streamlit as st
from docx import Document
from dotenv import load_dotenv
from openai import OpenAI

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv()

SYSTEM_HEADHUNTER = """你是具备全球化视野的专业全行业资深猎头顾问，熟悉互联网、金融、数据分析、产品与技术、运营与增长、创意与内容、专业服务、制造与供应链等多赛道与跨文化职场环境。
请先根据用户提供的岗位 JD 自行识别其所属细分行业与职能类型（例如互联网、金融、数据分析、运营、创意、研发、销售等），不要默认局限于公关或营销领域；所有判断须与 JD 实际语境一致。
你的风格：专业、直接、可执行；避免空话套话；基于用户提供的简历与岗位材料做判断，若材料不足请明确说明假设与边界。"""


def inject_minimal_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #f4f6f9 0%, #eef2f6 100%);
        }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 960px; }
        h1 { font-weight: 600 !important; letter-spacing: -0.02em; color: #1e293b !important; }
        h2, h3 { font-weight: 600 !important; color: #334155 !important; }
        [data-testid="stSidebar"] {
            background: #e8edf2 !important;
            border-right: 1px solid #d1dae4 !important;
        }
        [data-testid="stSidebar"] .stMarkdown { color: #475569; }
        div[data-testid="stExpander"] { background: #ffffffcc; border: 1px solid #d8e0ea; border-radius: 8px; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
        .stTabs [data-baseweb="tab"] {
            background: #fff;
            border: 1px solid #d8e0ea !important;
            border-radius: 8px !important;
            color: #475569;
        }
        .stTabs [aria-selected="true"] {
            background: #dbeafe !important;
            border-color: #93c5fd !important;
            color: #1e3a5f !important;
        }
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


def ensure_session_state() -> None:
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""
    if "jd_text" not in st.session_state:
        st.session_state.jd_text = ""


def main() -> None:
    st.set_page_config(
        page_title="CareerVibe",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_minimal_css()
    ensure_session_state()

    st.title("CareerVibe")
    st.caption("极简求职助手 · 简历与岗位智能分析")

    with st.sidebar:
        st.markdown("### 材料上传")
        st.markdown("**简历**：支持 **PDF** 或 **Word（.docx）**。")
        resume_file = st.file_uploader(
            "简历文件",
            type=["pdf", "docx"],
            help="上传 PDF 或 .docx，任选其一",
            key="resume_file",
        )

        st.markdown("**岗位 JD**")
        st.caption("有 PDF 时优先使用文件；无文件时使用下方文字。")
        jd_file = st.file_uploader(
            "JD 文件（PDF，可选）",
            type=["pdf"],
            key="jd_pdf",
        )
        jd_paste = st.text_area(
            "或直接粘贴 JD 全文",
            height=160,
            placeholder="若未上传 JD 的 PDF，请在此粘贴岗位描述…",
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
            "PDF 文件"
            if jd_file is not None
            else ("文本框" if jd_paste.strip() else "—")
        )
        st.write("简历：" + ("✓ 已读取" if r_ok else "— 待上传或内容过短"))
        st.write("JD：" + ("✓ 已读取（" + jd_src + "）" if j_ok else "— 请上传 PDF 或填写文本"))

    tab_match, tab_interview, tab_debrief = st.tabs(
        ["岗位匹配分析", "面试通关指南", "面试复盘评分"]
    )

    client = get_client()

    with tab_match:
        st.subheader("岗位匹配分析")
        if len(st.session_state.resume_text) < 50 or len(st.session_state.jd_text) < 50:
            st.info("请先在侧边栏上传并解析简历与 JD（纯文本需达到一定长度）。")
        else:
            if st.button("生成匹配分析", key="btn_match"):
                with st.spinner("正在对比简历与 JD…"):
                    user_prompt = f"""请先根据下方「岗位 JD」判断岗位所属行业/职能赛道（在脑中完成即可，无需单独输出），再对比「简历」与 JD，完成三项任务并 **仅输出合法 JSON**（不要 Markdown 代码块）：

1) match_score：0-100 的整数匹配得分。综合评估维度须通用化：核心硬技能匹配度、关键软实力（沟通、领导力、结构化思维、学习敏捷度等）、行业/业务语境契合度、工具与方法栈、成果可验证性与职级适配等；按 JD 所属行业调整权重，不要套用单一行业模板。
2) must_fill_skills：字符串数组，列出候选人相对 JD 仍 **必须补足** 的技能/能力/证据（每条具体、可行动）。若 JD 未写清，请依据该岗位所在行业的通行标准与 JD 隐含要求推断，而非局限于公关或营销。
3) sustainability_analysis：一段中文分析（250-450 字）。须结合你所识别的行业，讨论：中长期行业前景与竞争格局；该职级在市场上的薪酬水平大致区间（务必简要说明地域、公司类型或职级假设）；岗位/技能折旧与可持续性风险；对候选人择业的可执行建议。分析须与 JD 行业一致，避免泛泛而谈。

JSON 键名必须为：match_score, must_fill_skills, sustainability_analysis

--- 简历 ---
{st.session_state.resume_text[:12000]}

--- 岗位 JD ---
{st.session_state.jd_text[:12000]}
"""
                    raw = chat_completion(
                        client,
                        SYSTEM_HEADHUNTER,
                        user_prompt,
                        json_mode=True,
                        max_tokens=2048,
                    )
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        st.error("模型返回非标准 JSON，请重试或更换模型。")
                        st.code(raw)
                    else:
                        st.session_state["match_result"] = data

            data = st.session_state.get("match_result")
            if isinstance(data, dict):
                score = data.get("match_score")
                skills = data.get("must_fill_skills") or []
                sustain = data.get("sustainability_analysis") or ""
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric("匹配得分", f"{score} / 100" if score is not None else "—")
                with c2:
                    st.markdown("#### 必补技能")
                    if isinstance(skills, list) and skills:
                        for s in skills:
                            st.markdown(f"- {s}")
                    else:
                        st.write("—")
                st.markdown("#### 行业前景、薪酬与可持续性")
                st.write(sustain or "—")

    with tab_interview:
        st.subheader("面试通关指南")
        if len(st.session_state.resume_text) < 50 or len(st.session_state.jd_text) < 50:
            st.info("请先在侧边栏上传并解析简历与 JD。")
        else:
            if st.button("生成 20 题面试指南", key="btn_interview"):
                with st.spinner("正在生成高频问题与个性化答案（可能需要几十秒）…"):
                    user_prompt = f"""先根据「岗位 JD」识别行业与岗位类型，再生成 **恰好 20 个** 该岗位在真实面试中常见的高频问题；问题必须与 JD 语境一致，不要局限于公关或营销。
覆盖维度应通用化并择优组合，例如：核心硬技能与工具栈、业务理解与场景题、数据与结果指标、协作与跨团队沟通、领导力/影响力、压力与冲突、合规与风险（如适用）、职业规划等。

对每个问题 i（i=1..20），在同一段落结构中输出：
**Q{{i}}** 问题正文
**简历深挖**：结合下面「简历」中可核验的事实/项目/数据，指出面试官可能追问的点（若无关联则写「简历中未体现相关点，建议补充案例：…」）。
**建议回答**：给出详细、可直接背诵调整的「个性化建议答案」（第一人称、具体、含 STAR 要素更佳）。

使用 Markdown，严格按 Q1…Q20 编号，不要省略任何问题。

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
                st.markdown(guide)

    with tab_debrief:
        st.subheader("面试复盘评分")
        notes = st.text_area(
            "粘贴面试后的复盘笔记或转写文字",
            height=240,
            placeholder="可包含：题目、你的回答要点、面试官反馈、自我感觉等…",
            key="debrief_notes",
        )
        if st.button("提交复盘并评分", key="btn_debrief"):
            if len(notes.strip()) < 30:
                st.warning("内容过短，请补充更多复盘细节以便评分。")
            else:
                with st.spinner("正在评分与总结…"):
                    user_prompt = f"""你将以具备全球化视野的全行业资深猎头视角，对以下「面试复盘/转写」进行专业评估；请结合候选人所述岗位语境（若有）给出通用化、可迁移的反馈。请用 Markdown 输出，包含以下小节（使用二级标题）：

## 百分制评分
给出 0-100 的整数分数，并用 2-4 句说明扣分主要原因与加分点。

## 优点总结
条列 3-8 条，具体、可引用复盘中的表述。

## 详细改进建议
条列 **所有** 你认为需要改进的点（可多于 10 条），每条包含：问题表现 → 改进方向 → 可练习的具体动作（例如改写话术、补充数据、调整结构）。

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
            st.markdown(debrief)


if __name__ == "__main__":
    main()

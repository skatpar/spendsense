"""SpendSense app. Run from the repo root:  streamlit run app.py

Three pages: Add expense, This month, Model monitoring. Everything is stored
in SQLite (data/spendsense.db). The look comes from .streamlit/config.toml
plus the CSS below; the HTML helpers only draw cards, badges and bars."""

# Lets "float | None" style type hints work on Python 3.9 too.
from __future__ import annotations

import html
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config, db, monitoring, ocr
from src.predict import load_model, model_problem, model_version, predict
from src.retrain import HISTORY_PATH, retrain
from src.train import train_and_save

st.set_page_config(page_title="SpendSense", page_icon=":material/account_balance_wallet:",
                   layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
h1, h2, h3, h4, p, li, label, input, textarea, button, td, th,
div[data-testid="stMarkdownContainer"] { font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif; }
[data-testid="stAppDeployButton"], .stDeployButton, #MainMenu, footer { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px; }
[data-testid="stSidebar"] { background: #0F172A; }
[data-testid="stSidebar"] * { color: #CBD5E1; }
[data-testid="stSidebarNav"] a { border-radius: 8px; margin: 2px 0; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: rgba(37, 99, 235, 0.28); }
[data-testid="stSidebarNav"] a[aria-current="page"] span { color: #FFFFFF; font-weight: 600; }
[data-testid="stSidebarNav"] a:hover { background: rgba(148, 163, 184, 0.14); }
div[class*="st-key-card_"] { background: #FFFFFF; border-radius: 14px !important; border: 1px solid #E5E9F0 !important;
  padding: 20px 22px !important; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); }
.ss-title { font-size: 1.85rem !important; font-weight: 700 !important; color: #0F172A !important; margin: 0 !important; letter-spacing: -0.02em; }
.ss-sub { color: #64748B !important; margin: 0.2rem 0 1.5rem 0 !important; font-size: 1rem !important; }
.ss-section { font-size: 0.75rem !important; font-weight: 600 !important; letter-spacing: 0.07em; text-transform: uppercase;
  color: #64748B !important; margin: 0 0 0.5rem 0 !important; }
.ss-kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin-bottom: 18px; }
.ss-kpi { background: #FFFFFF; border: 1px solid #E5E9F0; border-radius: 14px; padding: 18px 20px; }
.ss-kpi-label { color: #64748B; font-size: 0.82rem; font-weight: 500; }
.ss-kpi-value { color: #0F172A; font-size: 1.7rem; font-weight: 700; margin-top: 4px; letter-spacing: -0.01em; }
.ss-kpi-note { color: #94A3B8; font-size: 0.8rem; margin-top: 2px; }
.ss-badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
.ss-badge-lg { font-size: 1.15rem; padding: 6px 16px; }
.ss-pill { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }
.ss-ok { background: #DCFCE7; color: #166534; }
.ss-warn { background: #FEF3C7; color: #92400E; }
.ss-bad { background: #FEE2E2; color: #991B1B; }
.ss-info { background: #DBEAFE; color: #1E40AF; }
.ss-suggest { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin: 6px 0 14px 0; }
.ss-conf { color: #334155; font-size: 0.9rem; margin-bottom: 14px; }
.ss-conf b { font-size: 1.5rem; color: #0F172A; margin-right: 4px; }
.ss-bar-row { display: grid; grid-template-columns: 120px 1fr 64px; align-items: center; gap: 10px; margin: 9px 0; }
.ss-bar-name { color: #334155; font-size: 0.88rem; font-weight: 500; }
.ss-bar-track { background: #EEF2F7; border-radius: 999px; height: 8px; overflow: hidden; }
.ss-bar-fill { height: 8px; border-radius: 999px; }
.ss-bar-val { color: #475569; font-size: 0.85rem; text-align: right; font-variant-numeric: tabular-nums; }
.ss-empty { text-align: center; padding: 38px 10px; color: #64748B; }
.ss-empty b { display: block; color: #0F172A; font-size: 1.02rem; margin-bottom: 4px; }
.ss-banner { border-radius: 14px; padding: 16px 20px; margin-bottom: 18px; border: 1px solid; }
.ss-banner b { display: block; font-size: 1rem; margin-bottom: 2px; }
.ss-banner.ok { background: #F0FDF4; border-color: #BBF7D0; color: #166534; }
.ss-banner.bad { background: #FEF2F2; border-color: #FECACA; color: #991B1B; }
.ss-banner.info { background: #EFF6FF; border-color: #BFDBFE; color: #1E40AF; }
.ss-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.ss-table th { text-align: left; color: #64748B; font-weight: 500; font-size: 0.78rem; text-transform: uppercase;
  letter-spacing: 0.05em; padding: 8px 6px; border-bottom: 1px solid #E5E9F0; }
.ss-table td { padding: 10px 6px; border-bottom: 1px solid #F1F4F8; color: #0F172A; }
.ss-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
.ss-muted { color: #94A3B8; font-size: 0.8rem; }
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #F5F7FB; }
div[data-testid="stElementContainer"]:has(button[data-testid="stBaseButton-primary"]) { width: 100% !important; }
div.stButton:has(button[data-testid="stBaseButton-primary"]) { width: 100% !important; }
[data-testid="stBaseButton-primary"] { width: 100%; border-radius: 10px; font-weight: 600; padding: 0.6rem 1rem; }
[data-testid="stBaseButton-secondary"] { border-radius: 10px; }
</style>
"""


# ---------------- small HTML helpers ----------------

def _shade(hex_color: str, factor: float) -> str:
    """Darken a colour so badge text stays readable on its pale background."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#{:02x}{:02x}{:02x}".format(int(r * factor), int(g * factor), int(b * factor))


def badge(category: str, large: bool = False) -> str:
    color = config.CATEGORY_COLORS.get(category, "#64748B")
    size = " ss-badge-lg" if large else ""
    return (f'<span class="ss-badge{size}" style="background:{color}1f;color:{_shade(color, 0.62)}">'
            f'{html.escape(category)}</span>')


def bar_rows(rows: list[tuple[str, float, str, str]]) -> str:
    """Rows of (name, share 0..1, value text, colour) drawn as horizontal bars."""
    out = []
    for name, share, value, color in rows:
        out.append(f'<div class="ss-bar-row"><span class="ss-bar-name">{html.escape(name)}</span>'
                   f'<div class="ss-bar-track"><div class="ss-bar-fill" style="width:{share * 100:.1f}%;'
                   f'background:{color}"></div></div><span class="ss-bar-val">{value}</span></div>')
    return "".join(out)


def kpis(items: list[tuple[str, str, str]]) -> None:
    cards = "".join(f'<div class="ss-kpi"><div class="ss-kpi-label">{label}</div>'
                    f'<div class="ss-kpi-value">{value}</div><div class="ss-kpi-note">{note}</div></div>'
                    for label, value, note in items)
    st.markdown(f'<div class="ss-kpis">{cards}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<p class="ss-title">{title}</p><p class="ss-sub">{subtitle}</p>', unsafe_allow_html=True)


def section(label: str) -> None:
    st.markdown(f'<p class="ss-section">{label}</p>', unsafe_allow_html=True)


def empty_state(title: str, text: str) -> None:
    st.markdown(f'<div class="ss-empty"><b>{title}</b>{text}</div>', unsafe_allow_html=True)


def banner(kind: str, title: str, text: str) -> None:
    st.markdown(f'<div class="ss-banner {kind}"><b>{title}</b>{text}</div>', unsafe_allow_html=True)


def pkr(amount: float) -> str:
    return f"Rs {amount:,.0f}"


def card(name: str):
    """A white bordered panel. The key adds a CSS class (st-key-card_<name>) the stylesheet targets."""
    return st.container(border=True, key=f"card_{name}")


@st.cache_resource
def get_connection():
    return db.connect()


conn = get_connection()


# ---------------- Page 1: Add expense ----------------

FORM_KEYS = ["description", "amount", "date", "ocr_text", "source"]


def reset_form_if_saved() -> None:
    """Widgets can't be cleared after they are drawn, so a save sets a flag
    and the form is cleared at the start of the next run instead."""
    if st.session_state.pop("clear_form", False):
        for key in FORM_KEYS:
            st.session_state.pop(key, None)
    st.session_state.setdefault("date", date.today())
    st.session_state.setdefault("amount", 0.0)


def prefill_from_receipt(upload) -> None:
    """Run OCR on the photo and put what it found into the form fields."""
    with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix) as tmp:
        tmp.write(upload.getvalue())
        tmp.flush()
        result = ocr.read_receipt(tmp.name)
    lines = [ln.strip() for ln in result["text"].splitlines() if ln.strip() and ocr.is_readable(ln.strip())]
    # The first readable line of a receipt is usually the shop's name.
    st.session_state["description"] = lines[0] if lines else ""
    if result["total"] is not None:
        st.session_state["amount"] = float(result["total"])
    if result["date"] is not None:
        st.session_state["date"] = date.fromisoformat(result["date"])
    st.session_state["ocr_text"] = result["text"]
    st.session_state["source"] = "receipt"


def receipt_upload() -> None:
    upload = st.file_uploader("Scan a receipt (optional)", type=["jpg", "jpeg", "png"],
                              help="Total, date and shop name are read with OCR and filled in below.")
    if upload is not None and st.session_state.get("ocr_file") != upload.file_id:
        with st.spinner("Reading receipt..."):
            prefill_from_receipt(upload)
        st.session_state["ocr_file"] = upload.file_id
    if st.session_state.get("ocr_text"):
        with st.expander("Text read from the receipt"):
            st.text(st.session_state["ocr_text"])


def expense_form() -> tuple[str, float, date]:
    with card("form"):
        section("Expense details")
        receipt_upload()
        description = st.text_input("Description", key="description",
                                    placeholder="e.g. Careem ride to university")
        left, right = st.columns(2)
        amount = left.number_input("Amount (PKR)", min_value=0.0, step=50.0, key="amount")
        when = right.date_input("Date", key="date")
    return description, amount, when


def suggestion_card(pred: dict) -> None:
    status = ('<span class="ss-pill ss-warn">Needs your review</span>' if pred["needs_review"]
              else '<span class="ss-pill ss-ok">High confidence</span>')
    bars = bar_rows([(o["category"], o["confidence"], f"{o['confidence']:.0%}",
                      config.CATEGORY_COLORS[o["category"]]) for o in pred["top3"]])
    st.markdown(f'<div class="ss-suggest">{badge(pred["category"], large=True)}{status}</div>'
                f'<div class="ss-conf"><b>{pred["confidence"]:.0%}</b>confidence</div>'
                f'<p class="ss-section">Top 3</p>{bars}', unsafe_allow_html=True)


def save_panel(description: str, amount: float, when: date) -> None:
    with card("suggest"):
        section("Suggested category")
        if not description.strip():
            empty_state("Waiting for a description", "Type what you spent on, or scan a receipt.")
            return
        pred = predict(description, amount or None, when.isoformat())
        suggestion_card(pred)
        final = st.selectbox("Category", config.CATEGORIES, index=config.CATEGORIES.index(pred["category"]),
                             help="Keep the suggestion or pick the right one. Corrections teach the model.")
        if st.button("Save expense", type="primary"):
            expense = {"date": when.isoformat(), "description": description.strip(), "amount_pkr": amount,
                       "category": final, "source": st.session_state.get("source", "manual")}
            db.save_expense(conn, expense, pred, model_version())
            st.session_state["saved_message"] = f"Saved {description.strip()} · {pkr(amount)} · {final}"
            st.session_state["clear_form"] = True
            st.rerun()


def recent_table(expenses: pd.DataFrame) -> str:
    rows = "".join(f'<tr><td>{r.date:%d %b %Y}</td><td>{html.escape(r.description)}</td>'
                   f'<td>{badge(r.category)}</td><td class="num">{pkr(r.amount_pkr)}</td></tr>'
                   for r in expenses.itertuples())
    return ('<table class="ss-table"><tr><th>Date</th><th>Description</th><th>Category</th>'
            f'<th style="text-align:right">Amount</th></tr>{rows}</table>')


def page_add_expense() -> None:
    page_header("Add expense", "Describe a purchase or scan a receipt. SpendSense suggests the category; you decide.")
    reset_form_if_saved()
    if "saved_message" in st.session_state:
        st.toast(st.session_state.pop("saved_message"), icon=":material/check_circle:")
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        description, amount, when = expense_form()
    with right:
        save_panel(description, amount, when)
    recent = db.read_expenses(conn).head(5)
    if not recent.empty:
        with card("recent"):
            section("Recent expenses")
            st.markdown(recent_table(recent), unsafe_allow_html=True)


# ---------------- Page 2: This month ----------------

def month_options(expenses: pd.DataFrame) -> list[pd.Period]:
    today = pd.Timestamp(date.today()).to_period("M")
    months = set(expenses["date"].dt.to_period("M")) | {today}
    return sorted(months, reverse=True)


def category_breakdown(month: pd.DataFrame) -> None:
    totals = month.groupby("category")["amount_pkr"].sum().sort_values(ascending=False)
    rows = [(c, v / totals.max() if totals.max() else 0, pkr(v), config.CATEGORY_COLORS[c])
            for c, v in totals.items()]
    st.markdown(bar_rows(rows), unsafe_allow_html=True)


def daily_chart(month: pd.DataFrame, period: pd.Period) -> None:
    days = pd.date_range(period.start_time, period.end_time.normalize(), freq="D")
    daily = month.groupby(month["date"].dt.normalize())["amount_pkr"].sum().reindex(days, fill_value=0)
    chart = pd.DataFrame({"day": days.strftime("%d"), "PKR": daily.values})
    st.bar_chart(chart, x="day", y="PKR", color="#2563EB", height=260)


def transactions(month: pd.DataFrame) -> None:
    table = month[["id", "date", "description", "category", "amount_pkr", "source"]].copy()
    table["date"] = table["date"].dt.date
    event = st.dataframe(
        table, hide_index=True, on_select="rerun", selection_mode="multi-row", key="tx_table",
        column_order=["date", "description", "category", "amount_pkr", "source"],
        column_config={"date": "Date", "description": "Description", "category": "Category",
                       "amount_pkr": st.column_config.NumberColumn("Amount (PKR)", format="%.0f"),
                       "source": "Source"})
    chosen = event.selection.rows if event is not None else []
    if st.button(f"Delete selected ({len(chosen)})", disabled=not chosen, icon=":material/delete:"):
        for row in chosen:
            db.delete_expense(conn, int(table.iloc[row]["id"]))
        st.rerun()


def days_counted(period: pd.Period) -> int:
    """Days to average over: so far for the current month, all days for a past one."""
    if period == pd.Timestamp(date.today()).to_period("M"):
        return date.today().day
    return period.days_in_month


def page_this_month() -> None:
    expenses = db.read_expenses(conn)
    head, pick = st.columns([3, 1])
    with head:
        page_header("This month", "Where your money went, by category and by day.")
    period = pick.selectbox("Month", month_options(expenses), format_func=lambda p: p.strftime("%B %Y"))
    month = expenses[expenses["date"].dt.to_period("M") == period]
    if month.empty:
        with card("empty_month"):
            empty_state("No expenses in this month", "Saved expenses will show up here.")
        return
    top = month.groupby("category")["amount_pkr"].sum().idxmax()
    days_so_far = days_counted(period)
    kpis([("Total spent", pkr(month["amount_pkr"].sum()), period.strftime("%B %Y")),
          ("Transactions", f"{len(month)}", f"{month['source'].eq('receipt').sum()} from receipts"),
          ("Top category", top, pkr(month.loc[month["category"] == top, "amount_pkr"].sum())),
          ("Daily average", pkr(month["amount_pkr"].sum() / days_so_far), f"over {days_so_far} days")])
    left, right = st.columns([1, 1.3], gap="large")
    with left, card("categories"):
        section("Spending by category")
        category_breakdown(month)
    with right, card("daily"):
        section("Spending by day")
        daily_chart(month, period)
    with card("transactions"):
        section("Transactions · select rows to delete")
        transactions(month)


# ---------------- Page 3: Model monitoring ----------------

def health_banner(log: pd.DataFrame, metadata: dict) -> float | None:
    recommended, recent = monitoring.retrain_recommended(log, metadata["test_accuracy"])
    n = monitoring.RETRAIN_MIN_ROWS
    if recent is None:
        banner("info", "Collecting data", f"{len(log)} of {n} predictions logged. The health check starts at {n}.")
    elif recommended:
        banner("bad", "Retraining recommended",
               f"Accuracy over the last {n} predictions is {recent:.0%}, more than 10 points below "
               f"the test accuracy of {metadata['test_accuracy']:.0%}.")
    else:
        banner("ok", "Model is healthy", f"The last {n} predictions were {recent:.0%} accurate "
               f"(test accuracy {metadata['test_accuracy']:.0%}).")
    return recent


def monitoring_kpis(log: pd.DataFrame, metadata: dict, recent: float | None) -> None:
    review = (log["confidence"] < config.CONFIDENCE_THRESHOLD).mean()
    kpis([("Live accuracy", f"{recent:.0%}" if recent is not None else "–",
           f"last {monitoring.RETRAIN_MIN_ROWS} predictions"),
          ("Test accuracy", f"{metadata['test_accuracy']:.1%}", "held-out test split"),
          ("Predictions logged", f"{len(log)}", f"{int(log['corrected'].sum())} corrected by you"),
          ("Needed review", f"{review:.0%}", f"confidence below {config.CONFIDENCE_THRESHOLD}")])


def weekly_charts(weekly: pd.DataFrame) -> None:
    # Grouped bars rather than lines: they stay readable with only one or two weeks of data.
    chart = weekly.assign(week=weekly["week"].dt.strftime("w/c %d %b"))
    left, right = st.columns(2, gap="large")
    with left, card("accuracy"):
        section("Accuracy per week (1 − corrected share)")
        st.bar_chart(chart, x="week", y=["accuracy", "corrected_share"], color=["#2563EB", "#F97316"],
                     stack=False, height=260)
    with right, card("confidence"):
        section("Confidence per week")
        st.bar_chart(chart, x="week", y=["avg_confidence", "needed_review_share"],
                     color=["#2563EB", "#F97316"], stack=False, height=260)


def mix_chart(log: pd.DataFrame, metadata: dict) -> None:
    mix = monitoring.category_mix(log, metadata["label_distribution"]).rename_axis("category").reset_index()
    with card("mix"):
        section("Predicted category mix vs training data")
        st.bar_chart(mix, x="category", y=["predicted (live)", "training data"], color=["#2563EB", "#CBD5E1"],
                     stack=False, horizontal=True, height=340)


def retrain_panel() -> None:
    with card("weekly"):
        section("Retraining")
        st.markdown("Adds your confirmed and corrected expenses to the training data, holds out the latest 20% "
                    "of them for testing, and keeps the new model only if macro-F1 does not drop.")
        if st.button("Retrain with my corrections", icon=":material/model_training:"):
            with st.spinner("Retraining... this takes about a minute"):
                result = retrain(conn)
            st.success("New model saved." if result.get("kept") else f"Kept the current model: {result.get('reason', 'macro-F1 would drop.')}")
            st.json(result, expanded=False)
        if HISTORY_PATH.exists():
            st.dataframe(pd.read_csv(HISTORY_PATH), hide_index=True)


def page_monitoring() -> None:
    _, metadata = load_model()
    page_header("Model monitoring", f"{metadata['model_name']} · version {metadata['model_version']} · "
                f"trained on {metadata['train_rows']:,} expenses")
    log = db.read_log(conn)
    if log.empty:
        with card("retrain"):
            empty_state("No predictions yet", "Every saved expense is logged here with the model's guess and your choice.")
        return
    recent = health_banner(log, metadata)
    monitoring_kpis(log, metadata, recent)
    weekly = monitoring.weekly_metrics(log)
    weekly_charts(weekly)
    mix_chart(log, metadata)
    with card("empty_log"):
        section("Weekly numbers")
        st.dataframe(weekly.assign(week=weekly["week"].dt.date).round(3), hide_index=True)
    retrain_panel()


# ---------------- Setup screen (only when the model can't be used) ----------------

def page_setup(problem: str) -> None:
    page_header("Set up SpendSense", "One step before the first expense.")
    with card("setup"):
        banner("info", "The model needs to be built on this computer", html.escape(problem))
        st.markdown("This trains the model from the data in `data/raw/` with the same settings as "
                    "notebook 03 (5-fold cross-validation) and saves it to `models/`. It takes about a minute.")
        if st.button("Build model now", type="primary", icon=":material/build:"):
            with st.spinner("Training... computing sentence embeddings and cross-validating"):
                metadata = train_and_save()
            st.success(f"Model ready: {metadata['model_name']}, test accuracy {metadata['test_accuracy']:.1%}.")
            st.rerun()


# ---------------- Navigation ----------------

st.markdown(CSS, unsafe_allow_html=True)
st.logo(str(config.ROOT / "assets" / "logo.svg"), size="large",
        icon_image=str(config.ROOT / "assets" / "logo_mark.svg"))
problem = model_problem()
if problem:
    page_setup(problem)
else:
    pages = [st.Page(page_add_expense, title="Add expense", icon=":material/add_circle:", url_path="add", default=True),
             st.Page(page_this_month, title="This month", icon=":material/calendar_month:", url_path="month"),
             st.Page(page_monitoring, title="Model monitoring", icon=":material/monitoring:", url_path="monitoring")]
    navigation = st.navigation(pages)
    st.sidebar.caption(f"Model {model_version()} · review below {config.CONFIDENCE_THRESHOLD:.0%} confidence")
    navigation.run()

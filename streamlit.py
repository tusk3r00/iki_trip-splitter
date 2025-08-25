import os
from io import BytesIO
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st

from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    ForeignKey, select, func
)
from sqlalchemy.orm import declarative_base, Session, relationship

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
st.set_page_config(page_title="Trip Splitter — standalone", page_icon="🧮", layout="wide")

DB_PATH = os.environ.get("TRIP_DB_PATH", os.path.join(os.getcwd(), "trip.db"))
engine = create_engine(f"sqlite:///{DB_PATH}", future=True, echo=False)
Base = declarative_base()
DEFAULT_N = 14

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
class Participant(Base):
    __tablename__ = "participants"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    position = Column(Integer, nullable=False, default=0)

class Expense(Base):
    __tablename__ = "expenses"
    id = Column(Integer, primary_key=True)
    description = Column(String, nullable=True)
    payer_id = Column(Integer, ForeignKey("participants.id"), nullable=False)
    amount = Column(Float, nullable=False, default=0.0)
    payer = relationship("Participant")
    consumers = relationship("Consumer", cascade="all, delete-orphan", back_populates="expense")

class Consumer(Base):
    __tablename__ = "consumers"
    expense_id = Column(Integer, ForeignKey("expenses.id"), primary_key=True)
    participant_id = Column(Integer, ForeignKey("participants.id"), primary_key=True)
    expense = relationship("Expense", back_populates="consumers")
    participant = relationship("Participant")

Base.metadata.create_all(engine)

# ------------------------------------------------------------
# Init DB
# ------------------------------------------------------------
@st.cache_resource
def _init_db():
    with Session(engine) as s:
        cnt = s.scalar(select(func.count(Participant.id)))
        if cnt == 0:
            for i in range(DEFAULT_N):
                s.add(Participant(name=f"Person {i+1}", position=i))
            s.commit()
    return True

_init_db()

# ------------------------------------------------------------
# Helpers: participants
# ------------------------------------------------------------
def get_participants():
    with Session(engine) as s:
        return s.scalars(select(Participant).order_by(Participant.position, Participant.id)).all()

def participants_names():
    return [p.name for p in get_participants()]

def set_participants(names: list[str]):
    if len(set(names)) != len(names):
        raise ValueError("Names must be unique")
    with Session(engine) as s:
        parts = list(s.scalars(select(Participant).order_by(Participant.position, Participant.id)).all())
        for pos in range(min(len(parts), len(names))):
            p = parts[pos]
            p.name = names[pos]
            p.position = pos
        if len(names) > len(parts):
            for pos in range(len(parts), len(names)):
                s.add(Participant(name=names[pos], position=pos))
        if len(parts) > len(names):
            for pos in range(len(names), len(parts)):
                p = parts[pos]
                # dacă există legături la cheltuieli, nu ștergem
                used_as_payer = s.scalar(select(func.count(Expense.id)).where(Expense.payer_id == p.id))
                used_as_consumer = s.scalar(select(func.count(Consumer.participant_id)).where(Consumer.participant_id == p.id))
                if not (used_as_payer or used_as_consumer):
                    s.delete(p)
        s.commit()

# ------------------------------------------------------------
# Helpers: expenses
# ------------------------------------------------------------
def load_expenses_df() -> pd.DataFrame:
    parts = get_participants()
    names = [p.name for p in parts]
    with Session(engine) as s:
        rows = []
        for ex in s.scalars(select(Expense)).all():
            row = {
                "id": ex.id,
                "description": ex.description or "",
                "payer": s.get(Participant, ex.payer_id).name,
                "amount": ex.amount,
            }
            flags = {c.participant_id for c in ex.consumers}
            for idx, p in enumerate(parts):
                row[f"c_{idx}"] = p.id in flags
            rows.append(row)
    cols = ["id", "description", "payer", "amount"] + [f"c_{i}" for i in range(len(names))]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

def save_expenses_df(df: pd.DataFrame):
    parts = get_participants()
    name_to_id = {p.name: p.id for p in parts}
    with Session(engine) as s:
        s.query(Consumer).delete()
        s.query(Expense).delete()
        s.commit()
        if df is None or df.empty:
            return
        for _, r in df.iterrows():
            payer = str(r.get("payer", ""))
            if payer not in name_to_id:
                continue
            amount = float(r.get("amount") or 0)
            if amount < 0: amount = 0.0
            ex = Expense(
                description=str(r.get("description", ""))[:255],
                payer_id=name_to_id[payer],
                amount=amount,
            )
            s.add(ex)
            s.flush()
            consumers_idx = [i for i in range(len(parts)) if bool(r.get(f"c_{i}", False))]
            for i in consumers_idx:
                s.add(Consumer(expense_id=ex.id, participant_id=parts[i].id))
        s.commit()

# ------------------------------------------------------------
# Core computation
# ------------------------------------------------------------
def compute(expenses_df: pd.DataFrame, names: list[str]):
    if expenses_df is None or expenses_df.empty:
        idx = names
        mat = pd.DataFrame(0.0, index=idx, columns=names)
        totals = pd.DataFrame({"paid": 0.0, "consumed": 0.0, "net": 0.0}, index=idx)
        settlements = pd.DataFrame(columns=["from", "to", "amount"])
        return mat, totals, settlements

    n = len(names)
    paid = {nm: 0.0 for nm in names}
    consumed = {nm: 0.0 for nm in names}
    mat = pd.DataFrame(0.0, index=names, columns=names)

    for _, r in expenses_df.iterrows():
        payer = str(r.get("payer", ""))
        amount = float(r.get("amount") or 0)
        if payer not in names or amount <= 0:
            continue
        consumers_idx = [i for i in range(n) if bool(r.get(f"c_{i}", False))]
        paid[payer] += amount
        if not consumers_idx:
            continue
        per = amount / len(consumers_idx)
        for i in consumers_idx:
            nm = names[i]
            consumed[nm] += per
            if nm != payer:
                mat.loc[nm, payer] += per

    totals = pd.DataFrame({"paid": pd.Series(paid), "consumed": pd.Series(consumed)})
    totals["net"] = totals["paid"] - totals["consumed"]
    settlements = minimize_transfers(totals["net"].to_dict())
    return mat.round(2), totals.round(2), settlements

def minimize_transfers(net: dict[str, float], eps: float = 1e-9):
    creditors = [[k, v] for k, v in net.items() if v > eps]
    debtors = [[k, -v] for k, v in net.items() if v < -eps]
    creditors.sort(key=lambda x: x[1], reverse=True)
    debtors.sort(key=lambda x: x[1], reverse=True)
    i = j = 0
    transfers = []
    while i < len(debtors) and j < len(creditors):
        dn, da = debtors[i]
        cn, ca = creditors[j]
        pay = min(da, ca)
        transfers.append({"from": dn, "to": cn, "amount": round(pay, 2)})
        da -= pay; ca -= pay
        if da <= eps: i += 1
        else: debtors[i][1] = da
        if ca <= eps: j += 1
        else: creditors[j][1] = ca
    return pd.DataFrame(transfers)

def to_excel(expenses: pd.DataFrame, matrix: pd.DataFrame, totals: pd.DataFrame, settlements: pd.DataFrame) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as w:
        if expenses is not None and not expenses.empty:
            expenses.to_excel(w, sheet_name="Expenses", index=False)
        else:
            pd.DataFrame().to_excel(w, sheet_name="Expenses", index=False)
        matrix.to_excel(w, sheet_name="WhoOwesWhom")
        totals.to_excel(w, sheet_name="Totals")
        settlements.to_excel(w, sheet_name="Settlements", index=False)
    return out.getvalue()

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("🧮 Trip Splitter — standalone (SQLite)")

with st.sidebar:
    st.header("Participants")
    names = participants_names()
    count = st.number_input("Number of participants", 1, 50, value=len(names))
    if count > len(names):
        names = names + [f"Person {i+1}" for i in range(len(names), count)]
        set_participants(names)
    elif count < len(names):
        names = names[:count]
        set_participants(names)
    new_names = []
    for i, nm in enumerate(names):
        new_names.append(st.text_input(f"Name {i+1}", value=nm, key=f"p_{i}"))
    if new_names != names:
        if len(set(new_names)) != len(new_names):
            st.error("Names must be unique.")
        else:
            set_participants(new_names)
            names = participants_names()
    if st.button("Reset to sample 14"):
        set_participants([f"Person {i+1}" for i in range(14)])
        st.rerun()

st.subheader("1) Receipts")
expenses_df = load_expenses_df()
for i in range(len(names)):
    col = f"c_{i}"
    if col not in expenses_df.columns:
        expenses_df[col] = False

col_config = {
    "description": st.column_config.TextColumn("Description"),
    "payer": st.column_config.SelectboxColumn("Payer", options=names, required=True),
    "amount": st.column_config.NumberColumn("Amount", min_value=0.0, step=1.0),
}
for i, nm in enumerate(names):
    col_config[f"c_{i}"] = st.column_config.CheckboxColumn(nm, default=False)

edited = st.data_editor(
    expenses_df.drop(columns=[c for c in ["id"] if c in expenses_df.columns]),
    num_rows="dynamic",
    use_container_width=True,
    column_config=col_config,
    key="editor_expenses",
)

c0, c1, _ = st.columns([1,1,2])
with c0:
    if st.button("💾 Save to DB"):
        save_expenses_df(edited)
        st.success("Saved!")
with c1:
    if st.button("🔄 Reload from DB"):
        st.rerun()

st.subheader("2) Results")
mat, totals, settlements = compute(edited, names)

cL, cR = st.columns(2)
with cL:
    st.markdown("### Net per person (paid - consumed)")
    st.dataframe(totals)
with cR:
    st.markdown("### Suggested minimal transfers")
    if settlements.empty:
        st.info("No transfers yet")
    else:
        st.dataframe(settlements)

st.markdown("### Who owes whom (debtor → creditor)")
st.dataframe(mat)

st.subheader("3) Export")
excel_bytes = to_excel(edited, mat, totals, settlements)
st.download_button(
    "Download Excel report",
    data=excel_bytes,
    file_name="trip_splitter.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption(
    "Data is stored in a local SQLite file (trip.db). On Streamlit Cloud, "
    "it persists across reruns but may reset on redeploy."
)

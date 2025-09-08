Awesome brief! I went through your three files and here’s a focused review with concrete, minimal fixes to make your “NYC 311 Data Analytics Agent” robust, aligned with the assignment (2020–present), and capable of producing real charts from real SQL results.

---

# High-impact fixes (do these first)

## 1) Use the correct date range (2020–present) and avoid sampling bias

Right now you load only the first 100k rows from the 2010–present CSV and don’t filter by year. That can skew results and misses the assignment spec. In `setup_existing_data.py`, read the full file (or chunk) and filter to `created_date >= 2020-01-01` before writing to SQLite. Also add two handy derived columns:

* `closed_within_3_days` (boolean)
* `is_geocoded` (boolean: latitude & longitude are both present/valid)

Where to change: the database creation step.&#x20;

**Suggested edit (key lines only):**

```python
# Read CSV in chunks to avoid memory issues, filter to 2020+
chunks = pd.read_csv(self.csv_path, chunksize=100_000, low_memory=False)
frames = []
for ch in chunks:
    ch.columns = ch.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.lower()
    # parse dates safely
    for col in ['created_date','closed_date']:
        if col in ch.columns:
            ch[col] = pd.to_datetime(ch[col], errors='coerce')
    # filter to 2020+
    ch = ch[ch['created_date'] >= '2020-01-01']
    # derived cols
    if {'created_date','closed_date'}.issubset(ch.columns):
        ch['closed_within_3_days'] = (ch['closed_date'] - ch['created_date']).dt.days <= 3
    else:
        ch['closed_within_3_days'] = pd.NA

    if {'latitude','longitude'}.issubset(ch.columns):
        ch['is_geocoded'] = ch['latitude'].notna() & ch['longitude'].notna()
    else:
        ch['is_geocoded'] = pd.NA

    frames.append(ch)

df = pd.concat(frames, ignore_index=True)
```

Keep (and extend) your indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_complaints_type ON complaints(complaint_type);
CREATE INDEX IF NOT EXISTS idx_complaints_zip ON complaints(incident_zip);
CREATE INDEX IF NOT EXISTS idx_complaints_date ON complaints(created_date);
CREATE INDEX IF NOT EXISTS idx_complaints_borough ON complaints(borough);
CREATE INDEX IF NOT EXISTS idx_complaints_status ON complaints(status);
CREATE INDEX IF NOT EXISTS idx_closed3 ON complaints(closed_within_3_days);
CREATE INDEX IF NOT EXISTS idx_geocoded ON complaints(is_geocoded);
```

## 2) Make the agent return structured data for plotting (JSON), not just prose

Currently the agent logs/executes SQL but returns a formatted paragraph. In `sql_agent.py`, have `_check_query` actually return a **JSON payload** (e.g., `SQL_JSON:` prefix) so the Streamlit app can parse it for charts instead of showing a placeholder “Sample, 100” bar.

Where to change: `_check_query` and `_format_response`.&#x20;

**Suggested minimal change in `_check_query`:**

```python
# After result = self.db.run(query)
# result is a string table; better to re-run with pandas for JSON
import sqlite3, json
conn = sqlite3.connect(self.db_path)
try:
    df = pd.read_sql_query(query, conn)
    json_payload = df.to_dict(orient="records")
    response = f"SQL_JSON: {json.dumps(json_payload)[:15000]}"  # safety cut
finally:
    conn.close()
```

Keep your existing logging. Also **enforce safety**:

* Reject non-SELECT statements.
* Strip trailing semicolons/multiple statements.
* Auto-append `LIMIT 50` if the query has no `LIMIT` (unless the question asks for all).

Example preflight before executing:

```python
q = query.strip().rstrip(';')
if not q.lower().startswith("select"):
    q = "SELECT complaint_type, COUNT(*) AS count FROM complaints GROUP BY complaint_type ORDER BY count DESC LIMIT 10"
if " limit " not in q.lower():
    q += " LIMIT 50"
query = q
```

**In `_format_response`**, if you see `SQL_JSON:`, parse JSON, then (1) give a short natural-language summary with the LLM, and (2) also **echo a compact JSON** (the same) in the message so the UI can visualize it:

```python
if last_message.content.startswith("SQL_JSON:"):
    raw = last_message.content.replace("SQL_JSON: ","",1)
    try:
        records = json.loads(raw)
    except Exception:
        records = []
    # Keep your LLM summary as you have it...
    # Then return a message that includes both the NL summary and the JSON payload:
    combined = f"{formatted_response.content}\n\n[[DATA_JSON]]{json.dumps(records)[:15000]}[[/DATA_JSON]]"
    return {"messages":[AIMessage(content=combined)]}
```

This `[[DATA_JSON]]...[[/DATA_JSON]]` wrapper makes it easy for the Streamlit UI to extract the data cleanly.

## 3) Actually plot real results in Streamlit (remove the placeholder)

Right now, charts are created only if the prompt contains the word “top/most” and with a dummy single bar. Parse the agent’s final message for `[[DATA_JSON]]...[[/DATA_JSON]]`, build a DataFrame, and choose a sensible default chart (bar for 2 columns; pie for “share/percent” queries; line for time-series if a date column is present).

Where to change: in `app.py`, after `response = sql_agent.run(prompt)` and before storing message.&#x20;

**Suggested snippet (drop the placeholder logic):**

```python
import re, json
response_text = response
chart = None

m = re.search(r"\[\[DATA_JSON\]\](.+?)\[\[/DATA_JSON\]\]", response_text, flags=re.S)
if m:
    try:
        data = json.loads(m.group(1))
        if isinstance(data, list) and data:
            df = pd.DataFrame(data)
            # heuristic: choose chart type
            if 'created_date' in df.columns or 'date' in df.columns:
                chart = create_chart(df, "line")
            elif df.shape[1] >= 2 and df[df.columns[1]].dtype in ('int64','float64'):
                chart = create_chart(df, "bar")
            else:
                chart = create_chart(df, "pie")
            # Clean response text (hide the JSON blob)
            response_text = re.sub(r"\[\[DATA_JSON\]\].+?\[\[/DATA_JSON\]\]", "", response_text, flags=re.S)
    except Exception:
        pass

st.markdown(response_text)
if chart:
    st.plotly_chart(chart, use_container_width=True)
```

Also, consume the “Sample Questions” buttons. You already set `st.session_state.user_question` but don’t send it. Add at the top of the chat input block:

```python
if st.session_state.get("user_question"):
    prompt = st.session_state.pop("user_question")
else:
    prompt = st.chat_input("Ask a question about NYC 311 data...")
```

## 4) Prevent duplicate file handlers in Streamlit

Streamlit can re-import modules and add duplicate handlers. In `sql_agent.py`, guard the logger setup.&#x20;

```python
if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith('sql_commands.log')
           for h in sql_logger.handlers):
    sql_logger.addHandler(sql_handler)
```

---

# Nice-to-have improvements (polish & performance)

* **Force-tool usage reliability**: `tool_choice="required"` is good, but some LLMs ignore it. Your fallback is solid; keep it. Also sanitize queries to one statement, no PRAGMAs, no DDL/DML.
* **LIMIT auto-injection**: already suggested; this saves you from huge results crashing Streamlit.
* **Use the DataFrame path for SQL**: For charting, prefer `pandas.read_sql_query` → JSON, not the string table from `SQLDatabase.run()`.
* **Add canned skills** (fast answers without LLM):

  * Top N complaint types
  * % closed within 3 days (overall/by type)
  * Zip with most complaints
  * Proportion geocoded
    These can be quick `read_sql_query` helpers exposed as buttons; the LLM stays for free-form questions.
* **UX**: Show a “Results” table under the chart (e.g., `st.dataframe(df)`), and allow user to pick chart (bar/pie/line) via a small selectbox when JSON is detected.
* **Caching**: Wrap the sidebar “Quick Stats” queries with `@st.cache_data(ttl=600)` so they don’t hit SQLite every rerun. Your `@st.cache_resource` for the client is good.&#x20;
* **Validation**: If a user asks for “top 10,” ensure the generated SQL contains `LIMIT 10`. If they ask “percent/share,” encourage agent to compute percentages in SQL (or compute in Python after fetching grouped counts).
* **Time series**: For questions like “trend over months,” create `created_month = DATE(created_date)` in SQL and group by it.
* **Schema introspection**: You added a node that prints schema content back into the conversation. Consider keeping it in the internal trace only; it need not clutter the user output.&#x20;

---

# Quick sanity SQLs (you can wire as helpers or for the agent to emulate)

* Top 10 complaint types:

```sql
SELECT complaint_type, COUNT(*) AS count
FROM complaints
GROUP BY complaint_type
ORDER BY count DESC
LIMIT 10;
```

* % closed within 3 days for top 5 types:

```sql
WITH top_types AS (
  SELECT complaint_type
  FROM complaints
  GROUP BY complaint_type
  ORDER BY COUNT(*) DESC
  LIMIT 5
)
SELECT c.complaint_type,
       AVG(CASE WHEN c.closed_within_3_days THEN 1.0 ELSE 0 END)*100 AS pct_closed_3d
FROM complaints c
JOIN top_types t ON t.complaint_type = c.complaint_type
GROUP BY c.complaint_type
ORDER BY pct_closed_3d DESC;
```

* Zip with most complaints:

```sql
SELECT incident_zip, COUNT(*) AS cnt
FROM complaints
WHERE incident_zip IS NOT NULL AND TRIM(incident_zip) <> ''
GROUP BY incident_zip
ORDER BY cnt DESC
LIMIT 1;
```

* Proportion geocoded:

```sql
SELECT
  SUM(CASE WHEN is_geocoded THEN 1 ELSE 0 END)*1.0 / COUNT(*) AS proportion_geocoded
FROM complaints;
```

---

# Tiny code smells & tweaks

* `ToolNode` is created but never used; you can drop it (no harm, just tidy).&#x20;
* In `app.py` chart heuristics, your line chart title uses columns reversed (`x vs y` wording); no biggie, but after switching to real JSON you’ll likely rename axes anyway.&#x20;
* Make sure your cleaned col names match downstream expectations: you refer to `latitude`, `longitude`, `incident_zip`, `complaint_type`, `status`, `borough` consistently (you do).&#x20;
* Keep the LLM prompt crisp: “Never invent numbers; always run SQL; only SELECT” — you already do that, good.&#x20;

---

If you want, I can drop in exact patched versions of each file (or a unified diff) next. Otherwise, apply the snippets above and you’ll:

* meet the **2020–present** requirement,
* return **structured JSON** for real charts,
* handle **large data safely**, and
* significantly improve UX and evaluation quality.

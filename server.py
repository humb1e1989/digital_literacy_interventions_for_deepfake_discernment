"""
Behaviour tracking server for interventions_demo.html

Local dev:   python server.py                  → SQLite (tracking.db)
Production:  set DATABASE_URL env var           → PostgreSQL (Railway/Render)

Endpoints:
  GET  /                          serve the demo HTML
  GET  /images/<path>             serve images (when hosted here)
  GET  /config.js                 serve frontend config
  POST /api/session               start a session
  POST /api/log                   log page dwell event
  POST /api/result                log task result
  POST /api/interactions          log interaction batch
  GET  /api/data                  export page_events CSV
  GET  /api/data/results          export task_results CSV
  GET  /api/data/interactions     export interactions CSV
"""
import os, uuid, datetime, csv, io
from flask import Flask, request, jsonify, send_from_directory, Response

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = os.environ.get('DATABASE_URL', '')

# Railway sometimes provides postgres:// — psycopg2 needs postgresql://
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = 'postgresql' + DATABASE_URL[8:]

USE_PG = bool(DATABASE_URL)

if USE_PG:
    import psycopg2  # type: ignore  (production only; install psycopg2-binary on Railway)
    PH = '%s'
    def get_conn():
        return psycopg2.connect(DATABASE_URL)
else:
    import sqlite3
    DB_PATH = os.path.join(BASE_DIR, 'tracking.db')
    PH = '?'
    def get_conn():
        return sqlite3.connect(DB_PATH)

app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')


# ── Database helpers ────────────────────────────────────────────────────────────

def _q(sql):
    """Replace ? placeholders for the active database."""
    return sql.replace('?', PH)

def db_exec(sql, params=()):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(_q(sql), params)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def db_many(sql, rows):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.executemany(_q(sql), rows)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def db_all(sql, params=()):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(_q(sql), params)
        return cur.fetchall()
    finally:
        conn.close()


# ── Schema ─────────────────────────────────────────────────────────────────────

AUTO_PK = 'SERIAL PRIMARY KEY' if USE_PG else 'INTEGER PRIMARY KEY'

def init_db():
    db_exec('''CREATE TABLE IF NOT EXISTS sessions (
        id          TEXT PRIMARY KEY,
        condition   TEXT,
        started_at  TEXT,
        user_agent  TEXT
    )''')
    db_exec(f'''CREATE TABLE IF NOT EXISTS page_events (
        id          {AUTO_PK},
        session_id  TEXT,
        page        TEXT,
        entered_at  TEXT,
        duration_ms INTEGER,
        click_count INTEGER
    )''')
    db_exec(f'''CREATE TABLE IF NOT EXISTS task_results (
        id                    {AUTO_PK},
        session_id            TEXT,
        page                  TEXT,
        image_label           TEXT,
        is_real               INTEGER,
        detection_choice      TEXT,
        detection_choice_idx  INTEGER,
        is_correct            INTEGER,
        share_choice          TEXT,
        seen_before           INTEGER,
        reaction_time_ms      INTEGER,
        total_time_ms         INTEGER,
        attempts              INTEGER
    )''')
    db_exec(f'''CREATE TABLE IF NOT EXISTS interactions (
        id              {AUTO_PK},
        session_id      TEXT,
        page            TEXT,
        event_type      TEXT,
        element_type    TEXT,
        element_label   TEXT,
        x_pct           REAL,
        y_pct           REAL,
        time_on_page_ms INTEGER,
        ts              TEXT
    )''')


# ── CORS ───────────────────────────────────────────────────────────────────────

@app.after_request
def add_cors(r):
    r.headers['Access-Control-Allow-Origin']  = '*'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    r.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return r


# ── API ────────────────────────────────────────────────────────────────────────

@app.route('/api/session', methods=['POST', 'OPTIONS'])
def create_session():
    if request.method == 'OPTIONS': return '', 204
    d   = request.get_json(silent=True) or {}
    sid = str(uuid.uuid4())
    db_exec('INSERT INTO sessions VALUES (?,?,?,?)',
            (sid, d.get('condition', 'unknown'),
             datetime.datetime.now().isoformat(),
             request.headers.get('User-Agent', '')))
    return jsonify({'session_id': sid})

@app.route('/api/log', methods=['POST', 'OPTIONS'])
def log_event():
    if request.method == 'OPTIONS': return '', 204
    d = request.get_json(force=True, silent=True) or {}
    db_exec('INSERT INTO page_events (session_id,page,entered_at,duration_ms,click_count) VALUES (?,?,?,?,?)',
            (d.get('session_id'), d.get('page'), d.get('entered_at'),
             int(d.get('duration_ms', 0)), int(d.get('click_count', 0))))
    return jsonify({'ok': True})

@app.route('/api/result', methods=['POST', 'OPTIONS'])
def log_result():
    if request.method == 'OPTIONS': return '', 204
    d = request.get_json(force=True, silent=True) or {}
    db_exec('''INSERT INTO task_results
               (session_id,page,image_label,is_real,detection_choice,
                detection_choice_idx,is_correct,share_choice,seen_before,
                reaction_time_ms,total_time_ms,attempts)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
            (d.get('session_id'), d.get('page'), d.get('image_label'),
             d.get('is_real'), d.get('detection_choice'), d.get('detection_choice_idx'),
             d.get('is_correct'), d.get('share_choice'), d.get('seen_before'),
             d.get('reaction_time_ms'), d.get('total_time_ms'), d.get('attempts')))
    return jsonify({'ok': True})

@app.route('/api/interactions', methods=['POST', 'OPTIONS'])
def log_interactions():
    if request.method == 'OPTIONS': return '', 204
    data = request.get_json(force=True, silent=True) or []
    if isinstance(data, dict): data = [data]
    db_many('''INSERT INTO interactions
               (session_id,page,event_type,element_type,element_label,
                x_pct,y_pct,time_on_page_ms,ts)
               VALUES (?,?,?,?,?,?,?,?,?)''',
            [(e.get('session_id'), e.get('page'), e.get('event_type'),
              e.get('element_type'), e.get('element_label'),
              e.get('x_pct'), e.get('y_pct'),
              e.get('time_on_page_ms'), e.get('ts'))
             for e in data])
    return jsonify({'ok': True, 'count': len(data)})


# ── CSV exports ────────────────────────────────────────────────────────────────

def _csv(rows, cols, filename):
    buf = io.StringIO()
    csv.writer(buf).writerows([cols] + [list(r) for r in rows])
    return Response(buf.getvalue().encode('utf-8'), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename={filename}'})

@app.route('/api/data')
def export_page_events():
    rows = db_all('''SELECT s.id,s.condition,s.started_at,
                            e.page,e.entered_at,e.duration_ms,e.click_count
                     FROM sessions s LEFT JOIN page_events e ON s.id=e.session_id
                     ORDER BY s.started_at,e.entered_at''')
    return _csv(rows, ['session_id','condition','session_started',
                       'page','entered_at','duration_ms','click_count'], 'page_events.csv')

@app.route('/api/data/results')
def export_results():
    rows = db_all('''SELECT session_id,page,image_label,is_real,
                            detection_choice,detection_choice_idx,is_correct,
                            share_choice,seen_before,reaction_time_ms,total_time_ms,attempts
                     FROM task_results ORDER BY id''')
    return _csv(rows, ['session_id','page','image_label','is_real',
                       'detection_choice','detection_choice_idx','is_correct',
                       'share_choice','seen_before','reaction_time_ms',
                       'total_time_ms','attempts'], 'task_results.csv')

@app.route('/api/data/interactions')
def export_interactions():
    rows = db_all('''SELECT session_id,page,event_type,element_type,element_label,
                            x_pct,y_pct,time_on_page_ms,ts
                     FROM interactions ORDER BY id''')
    return _csv(rows, ['session_id','page','event_type','element_type','element_label',
                       'x_pct','y_pct','time_on_page_ms','ts'], 'interactions.csv')


# ── Static files ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'interventions_demo.html')

@app.route('/config.js')
def serve_config():
    return send_from_directory(BASE_DIR, 'config.js')

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'images'), filename)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    init_db()
    mode = 'PostgreSQL' if USE_PG else f'SQLite ({DB_PATH})'
    print(f'\n  Tracking server — {mode}')
    print('  Demo:             http://localhost:5001/')
    print('  Page dwell CSV:   http://localhost:5001/api/data')
    print('  Task results CSV: http://localhost:5001/api/data/results')
    print('  Interactions CSV: http://localhost:5001/api/data/interactions\n')
    app.run(debug=False, port=int(os.environ.get('PORT', 5001)))

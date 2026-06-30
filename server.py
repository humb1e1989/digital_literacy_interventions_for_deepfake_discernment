"""
Behaviour tracking server for interventions_demo.html
Run:        python server.py
Demo:       http://localhost:5001/
Export:     http://localhost:5001/api/data            (page dwell + clicks)
            http://localhost:5001/api/data/results    (task choices + correctness)
            http://localhost:5001/api/data/interactions (every mouse event)
"""
from flask import Flask, request, jsonify, send_from_directory, Response
import sqlite3, uuid, datetime, csv, io, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, 'tracking.db')

app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')


# ── Database ───────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        # Session table
        conn.execute('''CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            condition   TEXT,
            started_at  TEXT,
            user_agent  TEXT
        )''')

        # Page-level dwell time + click count
        conn.execute('''CREATE TABLE IF NOT EXISTS page_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            page        TEXT,
            entered_at  TEXT,
            duration_ms INTEGER,
            click_count INTEGER
        )''')

        # Per-image task results (detection choice, share, seen, timing)
        conn.execute('''CREATE TABLE IF NOT EXISTS task_results (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id            TEXT,
            page                  TEXT,
            image_label           TEXT,
            is_real               INTEGER,  -- 1=real 0=fake NULL=n/a
            detection_choice      TEXT,
            detection_choice_idx  INTEGER,  -- 0=Def.real 1=Prob.real 2=Prob.fake 3=Def.fake
            is_correct            INTEGER,  -- 1/0/NULL
            share_choice          TEXT,
            seen_before           INTEGER,
            reaction_time_ms      INTEGER,  -- ms from image shown to first detection click
            total_time_ms         INTEGER,  -- ms from image shown to Next click
            attempts              INTEGER   -- gamified only: clicks before round ended
        )''')

        # Fine-grained interaction events
        conn.execute('''CREATE TABLE IF NOT EXISTS interactions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT,
            page            TEXT,
            event_type      TEXT,   -- click | mouseenter | mouseleave | mousemove | change
            element_type    TEXT,   -- scale-btn | share-btn | next-btn | nav-btn | image | checkbox | …
            element_label   TEXT,
            x_pct           REAL,   -- 0–1 fraction of viewport width  (NULL for hover)
            y_pct           REAL,   -- 0–1 fraction of viewport height (NULL for hover)
            time_on_page_ms INTEGER,
            ts              TEXT    -- ISO 8601 timestamp
        )''')

        conn.commit()


# ── CORS ───────────────────────────────────────────────────────────────────────

@app.after_request
def add_cors(r):
    r.headers['Access-Control-Allow-Origin']  = '*'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    r.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return r


# ── API: session ───────────────────────────────────────────────────────────────

@app.route('/api/session', methods=['POST', 'OPTIONS'])
def create_session():
    if request.method == 'OPTIONS': return '', 204
    data = request.get_json(silent=True) or {}
    sid  = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute('INSERT INTO sessions VALUES (?,?,?,?)',
            (sid, data.get('condition', 'unknown'),
             datetime.datetime.now().isoformat(),
             request.headers.get('User-Agent', '')))
        conn.commit()
    return jsonify({'session_id': sid})


# ── API: page-level dwell log ──────────────────────────────────────────────────

@app.route('/api/log', methods=['POST', 'OPTIONS'])
def log_event():
    if request.method == 'OPTIONS': return '', 204
    d = request.get_json(force=True, silent=True) or {}
    with get_db() as conn:
        conn.execute(
            'INSERT INTO page_events (session_id,page,entered_at,duration_ms,click_count) VALUES (?,?,?,?,?)',
            (d.get('session_id'), d.get('page'), d.get('entered_at'),
             int(d.get('duration_ms', 0)), int(d.get('click_count', 0))))
        conn.commit()
    return jsonify({'ok': True})


# ── API: task result (one row per image) ──────────────────────────────────────

@app.route('/api/result', methods=['POST', 'OPTIONS'])
def log_result():
    if request.method == 'OPTIONS': return '', 204
    d = request.get_json(force=True, silent=True) or {}
    with get_db() as conn:
        conn.execute(
            '''INSERT INTO task_results
               (session_id,page,image_label,is_real,
                detection_choice,detection_choice_idx,is_correct,
                share_choice,seen_before,reaction_time_ms,total_time_ms,attempts)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
            (d.get('session_id'), d.get('page'), d.get('image_label'),
             d.get('is_real'), d.get('detection_choice'), d.get('detection_choice_idx'),
             d.get('is_correct'), d.get('share_choice'), d.get('seen_before'),
             d.get('reaction_time_ms'), d.get('total_time_ms'), d.get('attempts')))
        conn.commit()
    return jsonify({'ok': True})


# ── API: batch interaction events ─────────────────────────────────────────────

@app.route('/api/interactions', methods=['POST', 'OPTIONS'])
def log_interactions():
    if request.method == 'OPTIONS': return '', 204
    data = request.get_json(force=True, silent=True) or []
    if isinstance(data, dict): data = [data]
    with get_db() as conn:
        conn.executemany(
            '''INSERT INTO interactions
               (session_id,page,event_type,element_type,element_label,
                x_pct,y_pct,time_on_page_ms,ts)
               VALUES (?,?,?,?,?,?,?,?,?)''',
            [(e.get('session_id'), e.get('page'), e.get('event_type'),
              e.get('element_type'), e.get('element_label'),
              e.get('x_pct'), e.get('y_pct'),
              e.get('time_on_page_ms'), e.get('ts'))
             for e in data])
        conn.commit()
    return jsonify({'ok': True, 'count': len(data)})


# ── Exports ────────────────────────────────────────────────────────────────────

def _csv_response(rows, columns, filename):
    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(columns)
    for row in rows: w.writerow(list(row))
    return Response(buf.getvalue().encode('utf-8'), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename={filename}'})

@app.route('/api/data')
def export_page_events():
    with get_db() as conn:
        rows = conn.execute('''
            SELECT s.id,s.condition,s.started_at,
                   e.page,e.entered_at,e.duration_ms,e.click_count
            FROM sessions s
            LEFT JOIN page_events e ON s.id=e.session_id
            ORDER BY s.started_at,e.entered_at''').fetchall()
    return _csv_response(rows,
        ['session_id','condition','session_started','page','entered_at','duration_ms','click_count'],
        'page_events.csv')

@app.route('/api/data/results')
def export_results():
    with get_db() as conn:
        rows = conn.execute('''
            SELECT session_id,page,image_label,is_real,
                   detection_choice,detection_choice_idx,is_correct,
                   share_choice,seen_before,reaction_time_ms,total_time_ms,attempts
            FROM task_results
            ORDER BY rowid''').fetchall()
    return _csv_response(rows,
        ['session_id','page','image_label','is_real',
         'detection_choice','detection_choice_idx','is_correct',
         'share_choice','seen_before','reaction_time_ms','total_time_ms','attempts'],
        'task_results.csv')

@app.route('/api/data/interactions')
def export_interactions():
    with get_db() as conn:
        rows = conn.execute('''
            SELECT session_id,page,event_type,element_type,element_label,
                   x_pct,y_pct,time_on_page_ms,ts
            FROM interactions
            ORDER BY rowid''').fetchall()
    return _csv_response(rows,
        ['session_id','page','event_type','element_type','element_label',
         'x_pct','y_pct','time_on_page_ms','ts'],
        'interactions.csv')


# ── Static files ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'interventions_demo.html')

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'images'), filename)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    init_db()
    print('\n  Tracking server ready')
    print('  Demo:              http://localhost:5001/')
    print('  Page dwell CSV:    http://localhost:5001/api/data')
    print('  Task results CSV:  http://localhost:5001/api/data/results')
    print('  Interactions CSV:  http://localhost:5001/api/data/interactions\n')
    app.run(debug=False, port=5001)

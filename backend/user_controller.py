import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session
from ai import *
import os

app = Flask(__name__)
app.secret_key = os.urandom(42)

DATABASE = 'not_yet_implemented.db'


def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS review (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            review TEXT NOT NULL,
            rating INTEGER,
            predicted_rating INTEGER,
            FOREIGN KEY (user_id) REFERENCES user(id)
        );
    """)

    conn.commit()
    conn.close()


@app.route('/')
def home():
    if 'user_id' in session:
        return render_template('review_form.html')
    else:
        return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO user (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return render_template('register.html', message="Username already exists.")
        conn.close()
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user WHERE username = ? AND password = ?", (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect(url_for('home'))
        else:
            return render_template('login.html', message="Invalid credentials.")
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


@app.route('/review', methods=['GET', 'POST'])
def review():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        review_text = request.form['review']
        rating = request.form.get('rating', None)
        try:
            rating = int(rating)
        except ValueError:
            rating = None
        user_id = session['user_id']
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        predicted_rating = get_prediction(review_text)
        cursor.execute(
            "INSERT INTO review (user_id, review, rating, predicted_rating) VALUES (?, ?, ?, ?)",
            (user_id, review_text, rating, predicted_rating)
        )
        conn.commit()
        conn.close()
        return redirect(url_for('review'))

    return render_template('review_form.html')


@app.route('/predictions')
def predictions():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, review, rating, predicted_rating FROM review WHERE user_id = ?",
        (user_id,)
    )
    reviews = cursor.fetchall()
    conn.close()

    results = []
    for review_row in reviews:
        review_id, text, true_rating, predicted_rating = review_row

        results.append({
            "text": text,
            "true_rating": true_rating if true_rating is not None else "N/A",
            "predicted_rating": predicted_rating
        })

    return render_template('predictions.html', results=results)


if __name__ == "__main__":
    # init_db()
    app.run(debug=True)

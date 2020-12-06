from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)

class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)
    completed = db.Column(db.Integer, default=0)
    rating = db.Column(db.Float, default=0)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Task %r>' % self.id

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        task_content = request.form['content']
        rating = give_rating(task_content)
        new_task = Todo(content=task_content,rating=rating)
        db.session.add(new_task)
        db.session.commit()
        tasks = Todo.query.order_by(Todo.date_created).all()
        return render_template('review_table.html', tasks = tasks)
            
    else:
        return render_template('index.html')


@app.route('/delete/<int:id>')
def delete(id):
    task_to_delete = Todo.query.get_or_404(id)

    try:
        db.session.delete(task_to_delete)
        db.session.commit()
        tasks = Todo.query.order_by(Todo.date_created).all()
        return render_template('review_table.html',tasks=tasks)
    except:
        return 'There was a problem deleting that review'


@app.route('/stats')
def plot_stats():
    tasks = Todo.query.order_by(Todo.date_created).all()
    ratings = []
    for task in tasks:
        ratings.append(task.rating)
    bins = []
    for i in range(5):
        bins.append(0)
    for rat in ratings:
        if(rat<=1):
            bins[0] += 1
        elif(rat<=2):
            bins[1] += 1
        elif(rat<=3):
            bins[2] += 1
        elif(rat<=4):
            bins[3] += 1
        else:
            bins[4] += 1
    labels = ['0-1','1-2','2-3','3-4','4-5']
    sizes = bins
    explode = (0, 0, 0, 0, 0.1)
    fig1, ax1 = plt.subplots(figsize = (40,40))
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.2f%%',shadow=True, startangle=120)
    plt.savefig('static/images/new_plot.jpg')
    return plt.show()








def give_rating(customer_review):
    dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    corpus = []
    for i in range(0, 1000):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
    review = re.sub('[^a-zA-Z]', ' ',customer_review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
    cv = CountVectorizer()
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values
    X_train = X[0:-1,:]
    Y_train = y
    Y_train = Y_train.reshape(-1,1)
    classifier = svm.SVR()
    classifier.fit(X_train, Y_train)
    X = X[-1,:]
    X = X.reshape(1,-1)
    pred = classifier.predict(X)
    prob = pred[0]
    if(prob<0):
        return 0
    elif(prob>1):
        return 5
    else:
        return prob*5


if __name__ == "__main__":
    app.run(debug=True)

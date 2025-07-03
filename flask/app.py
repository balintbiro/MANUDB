from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('manudb.html')

@app.route('/manudb', methods=['GET', 'POST'])
def manudb():
    result = None
    if request.method == 'POST':
        try:
            number = int(request.form['number'])
            result = 2 ** number
        except ValueError:
            result = 'Invalid input. Please enter an integer.'
    return render_template('manudb.html', result=result)

@app.route('/export')
def export():
    return render_template('export.html')

@app.route('/visualize')
def visualize():
    return render_template('visualize.html')

if __name__ == '__main__':
    app.run(debug=True)

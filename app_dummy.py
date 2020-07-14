# app.py
from flask import Flask, request, jsonify, render_template
from task_dummy import example
from rq import Queue
from worker import conn
from flask_socketio import SocketIO
from flask_socketio import join_room, leave_room
import time
from time import sleep
from change_30nt_to_59nt import turn_switch_to_toehold

app = Flask(__name__)
app.config["DEBUG"]= True
app.config['SECRET_KEY'] = 'secret!'
#socketio = SocketIO(app)
socketio = SocketIO(app, async_mode="threading")
q = Queue(connection=conn)

# check for valid content
def is_toehold_valid(seq):
    if (len(seq) != 59):
        return 0 # bad seq length
    else: 
        valid_chars = ['A', 'U', 'T', 'C', 'G']
        for char in seq:
            if char not in valid_chars:
                return 1 # bad character composition
        return 2 # all is fine

# implement rooms according to https://flask-socketio.readthedocs.io/en/latest/
@socketio.on('join')
def on_join(data):
    username = data['username']
    room = data['room']
    join_room(room)
    send(username + ' has entered the room.', room=room)

@socketio.on('leave')
def on_leave(data):
    username = data['username']
    room = data['room']
    leave_room(room)
    send(username + ' has left the room.', room=room)

# this will get more complicated
def process_sequences(seq):
    seq_potential = seq.splitlines()
    seqs_to_return = []
    for seq in seq_potential:
        if seq != '':
            if seq[0] != '>':
                seqs_to_return.append(seq)
    return seqs_to_return

@app.route("/", methods=["GET", "POST"])
@app.route('/index')
def index():
    errors = ""
    if request.method == "POST":
        seq = None
        try:
            seq = request.form["seq"]
        except:
            errors += "ERROR: %s is not a valid sequence." % seq


        if request.form['action'] == 'Rank candidate toeholds':
            long_gemome = seq.upper()
            if long_gemome is not None:
                result_of_check = is_toehold_valid(long_gemome)
                if (result_of_check == 1):
                    errors += "ERROR: Region contains invalid characters."
                    return render_template('index.html', errors=errors)
                else: # it's fine
                    return render_template('prediction_results.html', title='Redesign', seq=long_gemome)
        else:
            seqs = process_sequences(seq)
            seqs_valid = []
            for sequence in seqs:
                sequence = sequence.upper()
                if sequence is not None:
                    if (len(sequence) == 30):
                        sequence = turn_switch_to_toehold(sequence)
                    result_of_check = is_toehold_valid(sequence)
                    if (result_of_check == 0):
                        errors += "ERROR: %s is not 30 or 59 nt." % sequence
                        return render_template('index.html', errors=errors)
                    elif (result_of_check == 1):
                        errors += "ERROR: %s contains invalid characters." % sequence
                        return render_template('index.html', errors=errors)
                    else:
                        seqs_valid.append(sequence)
            if request.form['action'] == 'Predict ON and OFF values':
                #prediction_table = run_prediction(seqs_valid)
                result_on = 1
                result_off = 2
                return render_template('prediction_results.html', title='Prediction', seq=seq, result_on=result_on, result_off=result_off)
            elif request.form['action'] == "Redesign toehold sequence**":
                if (len(seqs_valid) > 5):
                    errors += "ERROR: Cannot redesign more than 5 sequences at a time."
                    return render_template('index.html', errors=errors)
                return render_template('redesign_results.html', title='Redesign', seq=seq)
     
    return render_template('index.html', errors=errors)

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    #app.run(threaded=True, port=5000)
    socketio.run(app)

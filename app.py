# app.py
from flask import Flask, request, jsonify, render_template, redirect, url_for, render_template_string
from predict_toeholds import run_prediction, optimize_sequence, predict_genome_best
from rq import Queue
from worker import conn
from flask_socketio import SocketIO
from flask_socketio import join_room, leave_room
import numpy as np
from rq.job import Job
from time import sleep
from change_30nt_to_59nt import turn_switch_to_toehold

app = Flask(__name__)
app.config["DEBUG"]= True
app.config['SECRET_KEY'] = 'secret!'
#socketio = SocketIO(app)
socketio = SocketIO(app, async_mode="threading")
q = Queue(connection=conn)

# code borrowed from https://gist.github.com/vulcan25/23cae415aafec35abad21f150015a7ef
@app.route('/redesign_result/<string:id>')
def redesign_result(id):
    job = Job.fetch(id, connection=conn)
    status = job.get_status()
    if status is 'failed':
        return render_template('job_failed.html')
    elif status in ['queued', 'started', 'deferred']:
        return render_template('redesign_loading.html', status=status, refresh=True)
    elif status == 'finished':
        result = job.result 
        # If this is a string, we can simply return it:
        return render_template('redesign_results.html', result=result)

@app.route('/prediction_result/<string:id>')
def prediction_result(id):
    job = Job.fetch(id, connection=conn)
    status = job.get_status()
    if status is 'failed':
        return render_template('job_failed.html')
    elif status in ['queued', 'started', 'deferred']:
        return render_template('generic_loading.html', status=status, refresh=True)
    elif status == 'finished':
        result = job.result 
        # If this is a string, we can simply return it:
        return render_template('prediction_results.html', table=result)

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

# check for valid content
def is_region_valid(seq):
    if (len(seq) < 30):
        return 0 # bad seq length
    elif (seq[0] == '>'):
        return 1 # fasta file
    else: 
        valid_chars = ['A', 'U', 'T', 'C', 'G']
        for char in seq:
            if char not in valid_chars:
                return 2 # bad character composition
        return 3 # all is fine

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

# A welcome message to test our server
@app.route("/", methods=["GET", "POST"])
def index():
    errors = ""
    if request.method == "POST":
        seq = None
        try:
            seq = request.form["seq"]
        except:
            errors += "ERROR: %s is not a valid sequence." % seq
        if request.form['action'] == 'Rank candidate toeholds':
            long_genome = seq.upper()
            if long_genome is not None:
                result_of_check = is_region_valid(long_genome)
                if (result_of_check == 0):
                    errors += "ERROR: Region must be at least 30 nucleotides long."
                    return render_template('index.html', errors2=errors)
                elif (result_of_check == 1): 
                    errors += "ERROR: Please enter genome and not fasta file."
                    return render_template('index.html', errors2=errors)
                elif (result_of_check == 2):
                    errors += "ERROR: Region contains invalid characters."
                    return render_template('index.html', errors2=errors)
                else: # it's fine
                    job = q.enqueue(predict_genome_best, kwargs={'arg': long_genome}, job_timeout=600)  # 10 mins
                    return redirect(url_for('prediction_result', id=job.id))
        else:
            seqs = process_sequences(seq)
            seqs_valid = []
            if (len(seqs) == 0):
                errors += "ERROR: No valid sequences entered."
                return render_template('index.html', errors1=errors)
            for sequence in seqs:
                sequence = sequence.upper()
                if sequence is not None:
                    if (len(sequence) == 30):
                        sequence = turn_switch_to_toehold(sequence)
                    result_of_check = is_toehold_valid(sequence)
                    if (result_of_check == 0):
                        errors += "ERROR: %s is not 30 or 59 nt." % sequence
                        return render_template('index.html', errors1=errors)
                    elif (result_of_check == 1):
                        errors += "ERROR: %s contains invalid characters." % sequence
                        return render_template('index.html', errors1=errors)
                    else:
                        seqs_valid.append(sequence)
            if request.form['action'] == 'Predict ON and OFF values':
                #if (len(seqs_valid) > 10): # deploy worker # decided to always deploy worker
                job = q.enqueue(run_prediction, kwargs={'arg': seqs_valid}, job_timeout=600)  # 10 mins
                return redirect(url_for('prediction_result', id=job.id))
                #prediction_table = run_prediction(seqs_valid)
                #return render_template('prediction_results.html', title='Prediction', table=prediction_table)
            elif request.form['action'] == "Redesign toehold sequence**":
                if (len(seqs_valid) > 1):
                    errors += "ERROR: Cannot redesign more than 1 sequence at a time."
                    return render_template('index.html', errors1=errors)
                job = q.enqueue(optimize_sequence, kwargs={'arg': seqs_valid}, job_timeout=600)  # 10 mins
                return redirect(url_for('redesign_result', id=job.id))
            
    return render_template('index.html', errors1=errors)


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
   # app.run(threaded=True, port=5000)
   socketio.run(app)

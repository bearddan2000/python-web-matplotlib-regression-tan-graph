from flask import Flask, render_template, request
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(
    __name__,
    instance_relative_config=False,
    template_folder="templates"
)

def create_graph():
    filename = 'demo'

    # random but consistant data
    lst = [2,9,4,6,4]
    func_ptr = np.tan
    N = len(lst) # number of data points
    t = np.linspace(0, 4*np.pi, N)
    f = 1.15247 # Optional!! Advised not to use
    data = 3.0*func_ptr(f*t+0.001) + 0.5 + np.array(lst) # create artificial data with noise

    guess_mean = np.mean(data)
    guess_std = 3*np.std(data)/(2**0.5)/(2**0.5)
    guess_phase = 0
    guess_freq = 1
    guess_amp = 1

    # we'll use this to plot our first estimate. This might already be good enough for you
    data_first_guess = guess_std*func_ptr(t+guess_phase) + guess_mean

    # Define the function to optimize, in this case, we want to minimize the difference
    # between the actual data and our "guessed" parameters
    optimize_func = lambda x: x[0]*func_ptr(x[1]*t+x[2]) + x[3] - data
    est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]

    # recreate the fitted curve using the optimized parameters
    data_fit = est_amp*func_ptr(est_freq*t+est_phase) + est_mean

    # recreate the fitted curve using the optimized parameters

    fine_t = np.arange(0,max(t),0.1)
    data_fit=est_amp*func_ptr(est_freq*fine_t+est_phase)+est_mean
    plt.clf()
    plt.plot(t, data, '.')
    plt.plot(t, data_first_guess, label='first guess')
    plt.plot(fine_t, data_fit, label='after fitting')
    plt.legend()
    plt.savefig(f'static/img/{filename}.png')

@app.route('/', methods=['GET'])
def getIndex():
    create_graph()
    return render_template("index.html")

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

if __name__ == "__main__":
    app.run(host ='0.0.0.0', port = 5000, debug = True)
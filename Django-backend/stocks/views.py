from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import TickerForm
# from .tiingo import get_meta_data, get_price_data
from .scrapper import *
from .predictior import *

def index(request):
    if request.method == 'POST':
        form = TickerForm(request.POST)
        if form.is_valid():
            ticker = request.POST['ticker']
            return HttpResponseRedirect(ticker)
    else:
        form = TickerForm()
    return render(request, 'index.html', {'form': form})

def ticker(reuqest, tid):
    context = {}
    context['ticker']= tid
    context['profile']= get_company_profile(tid)
    context['price']=get_price_data(tid)
    make_company_candle_char(tid)
    make_company_line_char(tid)
    plot_past_predictions(tid)
    plot_future_predictions(tid)
    return render(reuqest, 'ticker.html', context)  

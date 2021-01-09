from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse
from .forms import TickerForm
from .scrapper import *
from .predictior import *

def index(request):
    print(request,"\n\n\n")
    if request.method == 'POST':
        form = TickerForm(request.POST)
        if form.is_valid():
            ticker = request.POST['ticker']
            start_date= request.POST['start_date']
            end_date= request.POST['end_date']
            url = reverse('ticker',kwargs={'tid':ticker, 'range': start_date + "_to_"+ end_date})
            return HttpResponseRedirect(url)
    else:
        form = TickerForm()
    return render(request, 'index.html', {'form': form})
    
def ticker(request, tid, range):
    start = range[:10]
    end = range[14:]
    context = {}
    context['ticker']= tid
    context['profile']= get_company_profile(tid)
    context['price']=get_price_data(tid)
    make_company_candle_char(tid, start, end)
    make_company_line_char(tid, start, end)
    # plot_past_predictions(tid, start, end)
    # plot_future_predictions(tid, start)
    return render(request, 'ticker.html', context)  

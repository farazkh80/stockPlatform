from django import forms

class TickerForm(forms.Form):
    ticker = forms.CharField(widget=forms.TextInput(
        attrs={'class': 'symbol_field'}
    ), label='', max_length=5)
    start_date = forms.CharField(widget=forms.DateInput(
        attrs={'class': 'start_date_field'}
    ), label='', max_length=10)
    end_date = forms.CharField(widget=forms.DateInput(
        attrs={'class': 'end_date_field'}
    ), label='', max_length=10)




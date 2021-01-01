from django import forms

class TickerForm(forms.Form):
    ticker = forms.CharField(widget=forms.TextInput(
        attrs={'class': 'symbol_field'}
    ), label='', max_length=5)


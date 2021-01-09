
$(document).ready(function () {

    var candleChart = $("#candle_stick_chart");
    var lineChart = $("#line_chart");
    var pastPredictions = $("#past_predictions_chart");
    var futurePredictions = $("#future_predictions_chart");
    var candleChartBtn = $("#candle_chart_btn");
    var lineChartBtn = $("#line_chart_btn");
    var pastPredictionsChartBtn = $("#past_predictions_chart_btn");
    var futurePredictionsChartBtn = $("#future_predictions_chart_btn");
    var ldsRing = $(".lds-ring");

    $( ".start_date_field").datepicker({
        changeMonth: true,
        changeYear: true
    });
    $( ".end_date_field" ).datepicker({
        changeMonth: true,
        changeYear: true
    });
    $( ".start_date_field").datepicker( "option", "dateFormat", "yy-mm-dd");
    $( ".end_date_field" ).datepicker( "option", "dateFormat", "yy-mm-dd");


    ldsRing.hide();
    lineChart.hide();
    pastPredictions.hide();
    futurePredictions.hide();


    candleChartBtn.click(function () {

        candleChart.hide();
        lineChart.hide();
        pastPredictions.hide();
        futurePredictions.hide();
        ldsRing.show();

        setTimeout(function () {
            ldsRing.hide();
            candleChart.show();
        }, 4000);
    })

    lineChartBtn.click(function () {

        candleChart.hide();
        lineChart.hide();
        pastPredictions.hide();
        futurePredictions.hide();
        ldsRing.show();
        
        setTimeout(function () {
            ldsRing.hide();
            lineChart.show();
        }, 4000);

    })

    pastPredictionsChartBtn.click(function () {

        candleChart.hide();
        lineChart.hide();
        pastPredictions.hide();
        futurePredictions.hide();
        ldsRing.show();

        setTimeout(function () {
            ldsRing.hide();
            pastPredictions.show();
        }, 4000);
    });

    futurePredictionsChartBtn.click(function () {
        candleChart.hide();
        lineChart.hide();
        pastPredictions.hide();
        futurePredictions.hide();
        ldsRing.show();

        setTimeout(function () {
            ldsRing.hide();
            futurePredictions.show();
        }, 4000);

    });

    // temporary fix if it doesn't work change

    $(".submit_symbol_button_from_ticker").click(function() {
        console.log( $(".symbol_field").val())
        var url = "http://127.0.0.1:8000/" + $(".symbol_field").val() + "/" + $(".start_date_field").val() + "_to_" + $(".end_date_field").val();
        window.location.href = url;
    });




});



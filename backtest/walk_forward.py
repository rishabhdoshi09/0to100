from datetime import timedelta
from backtest.backtester import Backtester
from analytics.reporter import PerformanceReporter

def walk_forward_backtest(historical_data, initial_capital=1_000_000,
                          train_years=2, test_months=1, step_months=1,
                          use_llm=False):
    if not historical_data:
        return []
    all_dates = sorted(historical_data[next(iter(historical_data))].index)
    if len(all_dates) < 2:
        return []
    start_date = all_dates[0]
    end_date = all_dates[-1]
    current_start = start_date
    results = []
    while True:
        train_end = current_start + timedelta(days=train_years*365)
        test_end = train_end + timedelta(days=test_months*30)
        if test_end > end_date:
            break
        _train_data = {sym: df.loc[current_start:train_end] for sym, df in historical_data.items()}  # noqa: F841
        test_data = {sym: df.loc[train_end:test_end] for sym, df in historical_data.items()}
        bt = Backtester(test_data, initial_capital=initial_capital, use_llm=use_llm)
        result = bt.run()
        reporter = PerformanceReporter(result)
        metrics = reporter.get_summary()
        metrics['test_start'] = train_end.strftime('%Y-%m-%d')
        metrics['test_end'] = test_end.strftime('%Y-%m-%d')
        results.append(metrics)
        current_start += timedelta(days=step_months*30)
        if current_start > test_end:
            current_start = test_end + timedelta(days=1)
    return results

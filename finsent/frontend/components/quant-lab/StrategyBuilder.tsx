'use client';

// =============================================================================
// STRATEGY BUILDER - Drag-and-Drop Backtest Rule Constructor
// =============================================================================

import React, { useState, useCallback } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://finsent-backend.onrender.com';

interface Rule {
  id: string;
  indicator: string;
  operator: string;
  value: string | number;
  action: 'entry' | 'exit';
}

interface TearsheetData {
  ticker: string;
  total_return: number;
  cagr: number;
  max_drawdown: number;
  volatility: number;
  sharpe_ratio: number;
  calmar_ratio: number;
  omega_ratio: number | null;
  tail_ratio: number;
  win_rate: number;
  total_trades: number;
  equity_curve: number[];
  dates: string[];
  drawdown_curve: number[];
}

const INDICATORS = [
  { value: 'RSI', label: 'RSI (14)' },
  { value: 'Price', label: 'Price' },
  { value: 'SMA_20', label: 'SMA 20' },
  { value: 'SMA_50', label: 'SMA 50' },
  { value: 'SMA_200', label: 'SMA 200' },
  { value: 'MACD', label: 'MACD' },
];

const OPERATORS = [
  { value: '<', label: 'Less than' },
  { value: '>', label: 'Greater than' },
  { value: '<=', label: 'Less or equal' },
  { value: '>=', label: 'Greater or equal' },
  { value: 'crosses_above', label: 'Crosses above' },
  { value: 'crosses_below', label: 'Crosses below' },
];

const COMPARE_VALUES = [
  { value: '30', label: '30', group: 'RSI' },
  { value: '50', label: '50', group: 'RSI' },
  { value: '70', label: '70', group: 'RSI' },
  { value: 'SMA_20', label: 'SMA 20', group: 'MA' },
  { value: 'SMA_50', label: 'SMA 50', group: 'MA' },
  { value: 'SMA_200', label: 'SMA 200', group: 'MA' },
  { value: 'EMA_12', label: 'EMA 12', group: 'MA' },
];

// Rule Block Component
const RuleBlock: React.FC<{
  rule: Rule;
  onUpdate: (rule: Rule) => void;
  onRemove: () => void;
}> = ({ rule, onUpdate, onRemove }) => {
  return (
    <div className={`flex items-center gap-3 p-4 rounded-xl border ${rule.action === 'entry'
        ? 'bg-success-50 border-success-200'
        : 'bg-coral-50 border-coral-200'
      }`}>
      {/* Action badge */}
      <span className={`px-2 py-1 text-xs font-bold uppercase rounded ${rule.action === 'entry'
          ? 'bg-success-100 text-success-700'
          : 'bg-coral-100 text-coral-700'
        }`}>
        {rule.action}
      </span>

      {/* Indicator select */}
      <select
        value={rule.indicator}
        onChange={(e) => onUpdate({ ...rule, indicator: e.target.value })}
        className="px-3 py-2 bg-white border border-cream-200 rounded-lg text-sm"
      >
        {INDICATORS.map((ind) => (
          <option key={ind.value} value={ind.value}>{ind.label}</option>
        ))}
      </select>

      {/* Operator select */}
      <select
        value={rule.operator}
        onChange={(e) => onUpdate({ ...rule, operator: e.target.value })}
        className="px-3 py-2 bg-white border border-cream-200 rounded-lg text-sm"
      >
        {OPERATORS.map((op) => (
          <option key={op.value} value={op.value}>{op.label}</option>
        ))}
      </select>

      {/* Value select/input */}
      <select
        value={String(rule.value)}
        onChange={(e) => onUpdate({ ...rule, value: e.target.value })}
        className="px-3 py-2 bg-white border border-cream-200 rounded-lg text-sm"
      >
        {COMPARE_VALUES.map((val) => (
          <option key={val.value} value={val.value}>{val.label}</option>
        ))}
      </select>

      {/* Remove button */}
      <button
        onClick={onRemove}
        className="ml-auto p-2 text-obsidian-400 hover:text-coral-600 hover:bg-coral-50 rounded-lg transition-colors"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
};

// Tearsheet Display Component
const Tearsheet: React.FC<{ data: TearsheetData }> = ({ data }) => {
  const MetricRow: React.FC<{ label: string; value: string | number | null; highlight?: boolean }> =
    ({ label, value, highlight }) => (
      <div className={`flex justify-between py-2 ${highlight ? 'text-lg font-semibold' : 'text-sm'}`}>
        <span className="text-obsidian-500">{label}</span>
        <span className={`font-mono ${highlight ? 'text-obsidian-900' : 'text-obsidian-700'}`}>
          {value ?? '—'}
        </span>
      </div>
    );

  return (
    <div className="p-6 bg-white/80 backdrop-blur-lg rounded-2xl border border-cream-200/50">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-electric-100 rounded-lg flex items-center justify-center">
          <span className="text-xl">📊</span>
        </div>
        <div>
          <h3 className="text-lg font-bold text-obsidian-900">Backtest Tearsheet</h3>
          <p className="text-sm text-obsidian-500">{data.ticker} • {data.total_trades} trades</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Returns */}
        <div className="space-y-1 border-r border-cream-200 pr-6">
          <p className="text-xs uppercase tracking-wide text-obsidian-400 mb-3">Returns</p>
          <MetricRow label="Total Return" value={`${data.total_return}%`} highlight />
          <MetricRow label="CAGR" value={`${data.cagr}%`} />
          <MetricRow label="Win Rate" value={`${data.win_rate}%`} />
        </div>

        {/* Risk */}
        <div className="space-y-1">
          <p className="text-xs uppercase tracking-wide text-obsidian-400 mb-3">Risk</p>
          <MetricRow label="Max Drawdown" value={`${data.max_drawdown}%`} highlight />
          <MetricRow label="Volatility" value={`${data.volatility}%`} />
          <MetricRow label="Sharpe Ratio" value={data.sharpe_ratio} />
        </div>

        {/* Advanced Ratios */}
        <div className="col-span-2 pt-4 border-t border-cream-200">
          <p className="text-xs uppercase tracking-wide text-obsidian-400 mb-3">Advanced Metrics</p>
          <div className="grid grid-cols-3 gap-4">
            <MetricRow label="Calmar Ratio" value={data.calmar_ratio} />
            <MetricRow label="Omega Ratio" value={data.omega_ratio} />
            <MetricRow label="Tail Ratio" value={data.tail_ratio} />
          </div>
        </div>
      </div>

      {/* Mini Equity Curve */}
      <div className="mt-6 pt-4 border-t border-cream-200">
        <p className="text-xs uppercase tracking-wide text-obsidian-400 mb-2">Equity Curve</p>
        <div className="h-24 flex items-end gap-px">
          {data.equity_curve.slice(-50).map((value, i, arr) => {
            const min = Math.min(...arr);
            const max = Math.max(...arr);
            const height = ((value - min) / (max - min)) * 100;
            return (
              <div
                key={i}
                className="flex-1 bg-electric-400 rounded-t"
                style={{ height: `${height}%` }}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
};

// Main Strategy Builder Component
export const StrategyBuilder: React.FC = () => {
  const [ticker, setTicker] = useState('SPY');
  const [period, setPeriod] = useState('5y');
  const [rules, setRules] = useState<Rule[]>([
    { id: '1', indicator: 'RSI', operator: '<', value: '30', action: 'entry' },
    { id: '2', indicator: 'RSI', operator: '>', value: '70', action: 'exit' },
  ]);
  const [loading, setLoading] = useState(false);
  const [tearsheet, setTearsheet] = useState<TearsheetData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const addRule = (action: 'entry' | 'exit') => {
    setRules([...rules, {
      id: Date.now().toString(),
      indicator: 'RSI',
      operator: action === 'entry' ? '<' : '>',
      value: action === 'entry' ? '30' : '70',
      action,
    }]);
  };

  const updateRule = (id: string, updated: Rule) => {
    setRules(rules.map(r => r.id === id ? updated : r));
  };

  const removeRule = (id: string) => {
    setRules(rules.filter(r => r.id !== id));
  };

  const runBacktest = useCallback(async () => {
    if (!ticker || rules.length === 0) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/quant/backtest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker: ticker.toUpperCase(),
          period,
          rules: rules.map(({ indicator, operator, value, action }) => ({
            indicator,
            operator,
            value: isNaN(Number(value)) ? value : Number(value),
            action,
          })),
        }),
      });

      if (!response.ok) throw new Error('Backtest failed');

      const data = await response.json();
      setTearsheet(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [ticker, period, rules]);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-obsidian-900">Strategy Forge</h2>
        <p className="text-obsidian-500">Build and backtest trading strategies with visual logic blocks</p>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-4">
        <div className="flex-1">
          <label className="block text-sm font-medium text-obsidian-600 mb-1">Ticker</label>
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            className="w-full px-4 py-2 border border-cream-200 rounded-lg focus:ring-2 focus:ring-electric-500"
            placeholder="SPY"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-obsidian-600 mb-1">Period</label>
          <select
            value={period}
            onChange={(e) => setPeriod(e.target.value)}
            className="px-4 py-2 border border-cream-200 rounded-lg focus:ring-2 focus:ring-electric-500"
          >
            <option value="1y">1 Year</option>
            <option value="2y">2 Years</option>
            <option value="5y">5 Years</option>
            <option value="10y">10 Years</option>
          </select>
        </div>
      </div>

      {/* Rules */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-obsidian-900">Trading Rules</h3>
          <div className="flex gap-2">
            <button
              onClick={() => addRule('entry')}
              className="px-3 py-1.5 bg-success-100 text-success-700 text-sm font-medium rounded-lg hover:bg-success-200 transition-colors"
            >
              + Entry Rule
            </button>
            <button
              onClick={() => addRule('exit')}
              className="px-3 py-1.5 bg-coral-100 text-coral-700 text-sm font-medium rounded-lg hover:bg-coral-200 transition-colors"
            >
              + Exit Rule
            </button>
          </div>
        </div>

        {rules.map((rule) => (
          <RuleBlock
            key={rule.id}
            rule={rule}
            onUpdate={(updated) => updateRule(rule.id, updated)}
            onRemove={() => removeRule(rule.id)}
          />
        ))}
      </div>

      {/* Run Button */}
      <button
        onClick={runBacktest}
        disabled={loading || !ticker || rules.length === 0}
        className="w-full py-4 bg-obsidian-900 text-white font-semibold rounded-xl hover:bg-obsidian-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Running Backtest...
          </span>
        ) : (
          'Run Backtest'
        )}
      </button>

      {/* Error */}
      {error && (
        <div className="p-4 bg-coral-50 border border-coral-200 rounded-xl text-coral-700">
          {error}
        </div>
      )}

      {/* Results */}
      {tearsheet && <Tearsheet data={tearsheet} />}
    </div>
  );
};

export default StrategyBuilder;

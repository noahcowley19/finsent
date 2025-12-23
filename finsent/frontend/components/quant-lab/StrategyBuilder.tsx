'use client';

// =============================================================================
// STRATEGY BUILDER COMPONENT
// =============================================================================
// Interface for building trading strategies
//
// Location: frontend/components/quant-lab/StrategyBuilder.tsx
//
// =============================================================================

import React, { useState } from 'react';

export interface StrategyCondition {
  id: string;
  indicator: string;
  operator: string;
  value: string;
}

export interface Strategy {
  id?: string;
  name: string;
  symbol: string;
  entryConditions: StrategyCondition[];
  exitConditions: StrategyCondition[];
  initialCapital: number;
  positionSize: number;
}

export interface StrategyBuilderProps {
  onRunBacktest: (strategy: Strategy) => void;
  onSave?: (strategy: Strategy) => void;
  isRunning?: boolean;
  initialStrategy?: Partial<Strategy>;
}

const indicators = [
  { value: 'sma_20', label: 'SMA (20)' },
  { value: 'sma_50', label: 'SMA (50)' },
  { value: 'sma_200', label: 'SMA (200)' },
  { value: 'ema_12', label: 'EMA (12)' },
  { value: 'ema_26', label: 'EMA (26)' },
  { value: 'rsi', label: 'RSI (14)' },
  { value: 'macd', label: 'MACD' },
  { value: 'macd_signal', label: 'MACD Signal' },
  { value: 'price', label: 'Price' },
  { value: 'volume', label: 'Volume' },
];

const operators = [
  { value: 'crosses_above', label: 'Crosses Above' },
  { value: 'crosses_below', label: 'Crosses Below' },
  { value: 'greater_than', label: 'Greater Than' },
  { value: 'less_than', label: 'Less Than' },
  { value: 'equals', label: 'Equals' },
];

export const StrategyBuilder: React.FC<StrategyBuilderProps> = ({
  onRunBacktest,
  onSave,
  isRunning = false,
  initialStrategy,
}) => {
  const [strategy, setStrategy] = useState<Strategy>({
    name: initialStrategy?.name || 'My Strategy',
    symbol: initialStrategy?.symbol || 'AAPL',
    entryConditions: initialStrategy?.entryConditions || [
      { id: '1', indicator: 'sma_20', operator: 'crosses_above', value: 'sma_50' },
    ],
    exitConditions: initialStrategy?.exitConditions || [
      { id: '1', indicator: 'sma_20', operator: 'crosses_below', value: 'sma_50' },
    ],
    initialCapital: initialStrategy?.initialCapital || 10000,
    positionSize: initialStrategy?.positionSize || 100,
  });

  const addCondition = (type: 'entry' | 'exit') => {
    const newCondition: StrategyCondition = {
      id: Date.now().toString(),
      indicator: 'price',
      operator: 'greater_than',
      value: '0',
    };

    if (type === 'entry') {
      setStrategy((prev) => ({
        ...prev,
        entryConditions: [...prev.entryConditions, newCondition],
      }));
    } else {
      setStrategy((prev) => ({
        ...prev,
        exitConditions: [...prev.exitConditions, newCondition],
      }));
    }
  };

  const updateCondition = (
    type: 'entry' | 'exit',
    id: string,
    field: keyof StrategyCondition,
    value: string
  ) => {
    const key = type === 'entry' ? 'entryConditions' : 'exitConditions';
    setStrategy((prev) => ({
      ...prev,
      [key]: prev[key].map((c) => (c.id === id ? { ...c, [field]: value } : c)),
    }));
  };

  const removeCondition = (type: 'entry' | 'exit', id: string) => {
    const key = type === 'entry' ? 'entryConditions' : 'exitConditions';
    setStrategy((prev) => ({
      ...prev,
      [key]: prev[key].filter((c) => c.id !== id),
    }));
  };

  const renderConditions = (
    conditions: StrategyCondition[],
    type: 'entry' | 'exit'
  ) => (
    <div className="space-y-3">
      {conditions.map((condition, index) => (
        <div key={condition.id} className="flex items-center gap-2">
          {index > 0 && (
            <span className="text-caption text-neutral-500 w-10">AND</span>
          )}
          {index === 0 && <span className="w-10" />}

          <select
            value={condition.indicator}
            onChange={(e) => updateCondition(type, condition.id, 'indicator', e.target.value)}
            className="flex-1 h-10 px-3 bg-cream-50 border border-border-medium rounded-lg text-body-sm focus:outline-none focus:border-navy-500"
          >
            {indicators.map((ind) => (
              <option key={ind.value} value={ind.value}>
                {ind.label}
              </option>
            ))}
          </select>

          <select
            value={condition.operator}
            onChange={(e) => updateCondition(type, condition.id, 'operator', e.target.value)}
            className="w-40 h-10 px-3 bg-cream-50 border border-border-medium rounded-lg text-body-sm focus:outline-none focus:border-navy-500"
          >
            {operators.map((op) => (
              <option key={op.value} value={op.value}>
                {op.label}
              </option>
            ))}
          </select>

          <input
            type="text"
            value={condition.value}
            onChange={(e) => updateCondition(type, condition.id, 'value', e.target.value)}
            placeholder="Value or indicator"
            className="w-32 h-10 px-3 bg-cream-50 border border-border-medium rounded-lg text-body-sm focus:outline-none focus:border-navy-500"
          />

          {conditions.length > 1 && (
            <button
              onClick={() => removeCondition(type, condition.id)}
              className="p-2 text-neutral-400 hover:text-error-600 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      ))}

      <button
        onClick={() => addCondition(type)}
        className="flex items-center gap-2 text-body-sm text-navy-600 hover:text-navy-700 transition-colors"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
        </svg>
        Add Condition
      </button>
    </div>
  );

  return (
    <div className="bg-white rounded-xl border border-border-light p-6">
      <h3 className="font-heading font-semibold text-heading-sm text-navy-900 mb-6">
        Strategy Builder
      </h3>

      {/* Basic settings */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div>
          <label className="block text-body-sm font-medium text-navy-700 mb-2">
            Strategy Name
          </label>
          <input
            type="text"
            value={strategy.name}
            onChange={(e) => setStrategy((prev) => ({ ...prev, name: e.target.value }))}
            className="w-full h-10 px-3 bg-cream-50 border border-border-medium rounded-lg text-body-sm focus:outline-none focus:border-navy-500"
          />
        </div>
        <div>
          <label className="block text-body-sm font-medium text-navy-700 mb-2">
            Symbol
          </label>
          <input
            type="text"
            value={strategy.symbol}
            onChange={(e) => setStrategy((prev) => ({ ...prev, symbol: e.target.value.toUpperCase() }))}
            className="w-full h-10 px-3 bg-cream-50 border border-border-medium rounded-lg text-body-sm focus:outline-none focus:border-navy-500"
          />
        </div>
        <div>
          <label className="block text-body-sm font-medium text-navy-700 mb-2">
            Initial Capital
          </label>
          <input
            type="number"
            value={strategy.initialCapital}
            onChange={(e) => setStrategy((prev) => ({ ...prev, initialCapital: Number(e.target.value) }))}
            className="w-full h-10 px-3 bg-cream-50 border border-border-medium rounded-lg text-body-sm focus:outline-none focus:border-navy-500"
          />
        </div>
      </div>

      {/* Entry conditions */}
      <div className="mb-8">
        <h4 className="text-body-sm font-semibold text-navy-900 mb-4 flex items-center gap-2">
          <span className="w-6 h-6 rounded bg-success-100 text-success-600 flex items-center justify-center">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
            </svg>
          </span>
          Entry Conditions
        </h4>
        {renderConditions(strategy.entryConditions, 'entry')}
      </div>

      {/* Exit conditions */}
      <div className="mb-8">
        <h4 className="text-body-sm font-semibold text-navy-900 mb-4 flex items-center gap-2">
          <span className="w-6 h-6 rounded bg-error-100 text-error-600 flex items-center justify-center">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </span>
          Exit Conditions
        </h4>
        {renderConditions(strategy.exitConditions, 'exit')}
      </div>

      {/* Actions */}
      <div className="flex items-center justify-end gap-3 pt-6 border-t border-border-light">
        {onSave && (
          <button
            onClick={() => onSave(strategy)}
            className="px-4 py-2 bg-white border border-border-medium rounded-lg text-body-sm font-medium text-navy-700 hover:bg-cream-50 transition-colors"
          >
            Save Strategy
          </button>
        )}
        <button
          onClick={() => onRunBacktest(strategy)}
          disabled={isRunning}
          className="flex items-center gap-2 px-6 py-2 bg-terra-500 text-white font-medium rounded-lg hover:bg-terra-600 transition-colors disabled:opacity-50"
        >
          {isRunning ? (
            <>
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Running...
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Run Backtest
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default StrategyBuilder;
